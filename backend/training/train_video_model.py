#!/usr/bin/env python3
"""
Video Model Training Script
Multi-Modal Deepfake Detection System - Video Modality Training

This script trains deep learning models for video-based deepfake detection
using various temporal architectures including I3D, SlowFast, and X3D.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from pathlib import Path
from datetime import datetime
import wandb
from tqdm import tqdm
import json
import cv2

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loaders.video_dataloader import DeepfakeVideoDataset, get_video_transforms
from augmentations.video_augmentations import VideoAugmentationPipeline
from models.video_models import I3DModel, SlowFastModel, X3DModel, VideoTransformerModel
from utils.logger import get_logger
from utils.metrics import MetricsCalculator
from utils.checkpoint_manager import CheckpointManager
from utils.video_utils import extract_frames, detect_faces_in_video

logger = get_logger(__name__)

class VideoTrainer:
    """Trainer class for video deepfake detection models"""

    def __init__(self, config_path: str, resume_from_checkpoint: str = None):
        """
        Initialize trainer

        Args:
            config_path: Path to configuration file
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()

        # Setup reproducibility
        self._set_seed(self.config['reproducibility']['seed'])

        # Initialize components
        self.model = self._build_model()
        self.train_loader, self.val_loader, self.test_loader = self._build_data_loaders()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.criterion = self._build_loss_function()
        self.metrics_calculator = MetricsCalculator(self.config['evaluation']['metrics'])

        # Setup logging and checkpointing
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.config['checkpoints']['save_dir'],
            save_top_k=self.config['checkpoints']['save_top_k'],
            monitor_metric=self.config['checkpoints']['monitor_metric']
        )

        self._setup_logging()
        self.current_epoch = 0

        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)

    def _load_config(self, config_path: str) -> dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_device(self) -> torch.device:
        """Setup training device"""
        device_config = self.config['hardware']['device']

        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU")
        else:
            device = torch.device(device_config)

        return device

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.config['reproducibility']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    def _build_model(self) -> nn.Module:
        """Build the model based on configuration"""
        model_config = self.config['model']
        architecture = model_config['architecture']

        if architecture == 'i3d':
            model = I3DModel(
                num_classes=model_config['num_classes'],
                pretrained=model_config['pretrained'],
                dropout_rate=model_config['dropout_rate'],
                input_shape=model_config['input_shape']
            )
        elif architecture == 'slowfast':
            model = SlowFastModel(
                num_classes=model_config['num_classes'],
                pretrained=model_config['pretrained'],
                **model_config['alternatives']['slowfast']
            )
        elif architecture == 'x3d':
            model = X3DModel(
                variant=model_config['alternatives']['x3d']['variant'],
                num_classes=model_config['num_classes'],
                pretrained=model_config['pretrained']
            )
        elif architecture == 'video_transformer':
            model = VideoTransformerModel(
                num_classes=model_config['num_classes'],
                **model_config['alternatives']['video_transformer']
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        model = model.to(self.device)

        # Multi-GPU setup if available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)

        logger.info(f"Model built: {architecture}")
        return model

    def _build_data_loaders(self) -> tuple:
        """Build data loaders for training, validation, and testing"""
        dataset_config = self.config['dataset']
        training_config = self.config['training']

        # Get transforms
        train_transform = get_video_transforms(
            config=self.config['augmentation']['train'],
            input_shape=self.config['model']['input_shape'],
            is_training=True
        )

        val_transform = get_video_transforms(
            config=self.config['augmentation']['validation'],
            input_shape=self.config['model']['input_shape'],
            is_training=False
        )

        # Create datasets
        train_dataset = DeepfakeVideoDataset(
            data_root=dataset_config['data_root'],
            split='train',
            transform=train_transform,
            config=dataset_config
        )

        val_dataset = DeepfakeVideoDataset(
            data_root=dataset_config['data_root'],
            split='val',
            transform=val_transform,
            config=dataset_config
        )

        test_dataset = DeepfakeVideoDataset(
            data_root=dataset_config['data_root'],
            split='test',
            transform=val_transform,
            config=dataset_config
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )

        logger.info(f"Data loaders created - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        optimizer_config = self.config['training']['optimizer']
        optimizer_name = optimizer_config['name']

        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                **optimizer_config['params']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler"""
        scheduler_config = self.config['training']['scheduler']
        scheduler_name = scheduler_config['name']

        if scheduler_name == 'cosine_annealing_warm_restart':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                **scheduler_config['params']
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return scheduler

    def _build_loss_function(self) -> nn.Module:
        """Build loss function with auxiliary losses"""
        loss_config = self.config['training']['loss']

        # Main loss
        main_criterion = nn.CrossEntropyLoss(
            label_smoothing=loss_config['params'].get('label_smoothing', 0.0)
        )

        # Return composite loss function
        return VideoCriterion(
            main_criterion=main_criterion,
            auxiliary_losses=loss_config.get('auxiliary_losses', {}),
            device=self.device
        )

    def _setup_logging(self):
        """Setup logging and experiment tracking"""
        logging_config = self.config['logging']

        # Setup Wandb
        if logging_config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=logging_config['project_name'],
                name=logging_config['experiment_name'],
                config=self.config,
                tags=logging_config['wandb'].get('tags', [])
            )

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch_data in enumerate(progress_bar):
            if len(batch_data) == 2:
                inputs, targets = batch_data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            else:
                # Handle additional data like optical flow
                inputs, targets, aux_data = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                aux_data = {k: v.to(self.device) for k, v in aux_data.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Calculate loss
            if hasattr(self.criterion, 'forward_with_aux'):
                loss = self.criterion.forward_with_aux(outputs, targets, aux_data)
            else:
                loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config['training'].get('gradient_clip_value'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip_value']
                )

            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_total += targets.size(0)
            running_correct += (predicted == targets).sum().item()

            # Update progress bar
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*running_correct/running_total:.2f}%',
                    'LR': f'{current_lr:.6f}'
                })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * running_correct / running_total

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }

    def validate_epoch(self) -> dict:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                else:
                    inputs, targets, _ = batch_data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Store predictions and targets for metrics calculation
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Calculate metrics
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        metrics = self.metrics_calculator.calculate_metrics(
            predictions=all_predictions,
            targets=all_targets
        )

        metrics['loss'] = running_loss / len(self.val_loader)

        return metrics

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        best_metric = float('-inf')

        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate epoch
            if epoch % self.config['evaluation']['validation_frequency'] == 0:
                val_metrics = self.validate_epoch()

                # Log metrics
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val AUC: {val_metrics.get('auc_roc', 0):.4f}")

                # Log to wandb
                if wandb.run:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        **{f'val_{k}': v for k, v in val_metrics.items() if k != 'loss'}
                    })

                # Check for improvement and save model
                current_metric = val_metrics[self.config['evaluation']['monitor_metric']]
                if current_metric > best_metric:
                    best_metric = current_metric

                    # Save best model
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metrics=val_metrics,
                        is_best=True
                    )

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            if epoch % self.config['checkpoints']['save_frequency'] == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics if 'val_metrics' in locals() else {},
                    is_best=False
                )

        logger.info("Training completed!")

    def test(self):
        """Test the best model"""
        logger.info("Testing best model...")

        # Load best checkpoint
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in tqdm(self.test_loader, desc="Testing"):
                if len(batch_data) == 2:
                    inputs, targets = batch_data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                else:
                    inputs, targets, _ = batch_data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)

                all_predictions.append(probabilities.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # Calculate test metrics
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        test_metrics = self.metrics_calculator.calculate_metrics(
            predictions=all_predictions,
            targets=all_targets
        )

        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Save test results
        results_path = os.path.join(self.config['paths']['output_root'], 'test_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)

        return test_metrics

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")


class VideoCriterion(nn.Module):
    """Composite loss function for video models with auxiliary losses"""

    def __init__(self, main_criterion, auxiliary_losses, device):
        super().__init__()
        self.main_criterion = main_criterion
        self.auxiliary_losses = auxiliary_losses
        self.device = device

    def forward(self, outputs, targets):
        """Forward pass with main loss only"""
        return self.main_criterion(outputs, targets)

    def forward_with_aux(self, outputs, targets, aux_data=None):
        """Forward pass with auxiliary losses"""
        main_loss = self.main_criterion(outputs, targets)
        total_loss = main_loss

        # Add auxiliary losses if configured and data is available
        if self.auxiliary_losses.get('temporal_consistency', {}).get('enabled') and aux_data:
            temporal_loss = self._temporal_consistency_loss(outputs, aux_data)
            weight = self.auxiliary_losses['temporal_consistency']['weight']
            total_loss += weight * temporal_loss

        if self.auxiliary_losses.get('frame_level_loss', {}).get('enabled') and aux_data:
            frame_loss = self._frame_level_loss(outputs, targets, aux_data)
            weight = self.auxiliary_losses['frame_level_loss']['weight']
            total_loss += weight * frame_loss

        return total_loss

    def _temporal_consistency_loss(self, outputs, aux_data):
        """Calculate temporal consistency loss"""
        # Implement temporal consistency loss logic
        # This is a placeholder - implement based on your specific requirements
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _frame_level_loss(self, outputs, targets, aux_data):
        """Calculate frame-level loss"""
        # Implement frame-level loss logic
        # This is a placeholder - implement based on your specific requirements
        return torch.tensor(0.0, device=self.device, requires_grad=True)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train video deepfake detection model")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test_only', action='store_true',
                       help='Only run testing')

    args = parser.parse_args()

    # Create trainer
    trainer = VideoTrainer(
        config_path=args.config,
        resume_from_checkpoint=args.resume
    )

    if args.test_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()

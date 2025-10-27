#!/usr/bin/env python3
"""
Audio Model Training Script
Multi-Modal Deepfake Detection System - Audio Modality Training

This script trains deep learning models for audio-based deepfake detection
using various architectures including ECAPA-TDNN, Wav2Vec2, and RawNet.
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
import wandb
from tqdm import tqdm
import json
import librosa

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loaders.audio_dataloader import DeepfakeAudioDataset, get_audio_transforms
from augmentations.audio_augmentations import AudioAugmentationPipeline
from models.audio_models import ECAPATDNNModel, Wav2Vec2Model, RawNetModel, AudioTransformerModel
from utils.logger import get_logger
from utils.metrics import MetricsCalculator
from utils.checkpoint_manager import CheckpointManager
from utils.audio_utils import extract_voice_features, detect_voice_activity

logger = get_logger(__name__)

class AudioTrainer:
    """Trainer class for audio deepfake detection models"""

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

        if architecture == 'ecapa_tdnn':
            model = ECAPATDNNModel(
                num_classes=model_config['num_classes'],
                channels=model_config['channels'],
                kernel_sizes=model_config['kernel_sizes'],
                dilations=model_config['dilations'],
                attention_channels=model_config['attention_channels'],
                global_context=model_config['global_context']
            )
        elif architecture == 'wav2vec2':
            model = Wav2Vec2Model(
                model_name=model_config['alternatives']['wav2vec2']['model_name'],
                num_classes=model_config['num_classes'],
                freeze_feature_extractor=model_config['alternatives']['wav2vec2']['freeze_feature_extractor']
            )
        elif architecture == 'rawnet':
            model = RawNetModel(
                num_classes=model_config['num_classes'],
                **model_config['alternatives']['rawnet']
            )
        elif architecture == 'transformer':
            model = AudioTransformerModel(
                num_classes=model_config['num_classes'],
                **model_config['alternatives']['transformer']
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

        # Create datasets
        train_dataset = DeepfakeAudioDataset(
            data_root=dataset_config['data_root'],
            split='train',
            config=dataset_config
        )

        val_dataset = DeepfakeAudioDataset(
            data_root=dataset_config['data_root'],
            split='val',
            config=dataset_config
        )

        test_dataset = DeepfakeAudioDataset(
            data_root=dataset_config['data_root'],
            split='test',
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

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            **optimizer_config['params']
        )

        return optimizer

    def _build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler"""
        scheduler_config = self.config['training']['scheduler']

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            **scheduler_config['params']
        )

        return scheduler

    def _build_loss_function(self) -> nn.Module:
        """Build loss function with auxiliary losses"""
        loss_config = self.config['training']['loss']

        # Main loss
        main_criterion = nn.CrossEntropyLR(
            label_smoothing=loss_config['params'].get('label_smoothing', 0.0)
        )

        return AudioCriterion(
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

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
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
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
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

        best_metric = float('inf')  # For EER, lower is better

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
                          f"Val EER: {val_metrics.get('equal_error_rate', 0):.4f}")

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
                current_metric = val_metrics.get('equal_error_rate', float('inf'))
                if current_metric < best_metric:
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
            for inputs, targets in tqdm(self.test_loader, desc="Testing"):
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


class AudioCriterion(nn.Module):
    """Composite loss function for audio models with auxiliary losses"""

    def __init__(self, main_criterion, auxiliary_losses, device):
        super().__init__()
        self.main_criterion = main_criterion
        self.auxiliary_losses = auxiliary_losses
        self.device = device

    def forward(self, outputs, targets):
        """Forward pass with main loss and auxiliary losses"""
        main_loss = self.main_criterion(outputs, targets)
        total_loss = main_loss

        # Add auxiliary losses if configured
        if self.auxiliary_losses.get('prototypical_loss', {}).get('enabled'):
            proto_loss = self._prototypical_loss(outputs, targets)
            weight = self.auxiliary_losses['prototypical_loss']['weight']
            total_loss += weight * proto_loss

        if self.auxiliary_losses.get('contrastive_loss', {}).get('enabled'):
            contrastive_loss = self._contrastive_loss(outputs, targets)
            weight = self.auxiliary_losses['contrastive_loss']['weight']
            total_loss += weight * contrastive_loss

        return total_loss

    def _prototypical_loss(self, outputs, targets):
        """Calculate prototypical loss for speaker verification"""
        # Implement prototypical loss logic
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _contrastive_loss(self, outputs, targets):
        """Calculate contrastive loss"""
        # Implement contrastive loss logic
        return torch.tensor(0.0, device=self.device, requires_grad=True)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train audio deepfake detection model")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test_only', action='store_true',
                       help='Only run testing')

    args = parser.parse_args()

    # Create trainer
    trainer = AudioTrainer(
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

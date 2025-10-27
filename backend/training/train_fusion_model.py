#!/usr/bin/env python3
"""
Fusion Model Training Script
Multi-Modal Deepfake Detection System - Multi-Modal Fusion Training

This script trains multi-modal fusion models that combine visual, temporal, 
and audio features for enhanced deepfake detection performance.
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
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_loaders.multimodal_dataloader import MultimodalDataset
from augmentations.multimodal_augmentations import MultimodalAugmentationPipeline
from models.fusion_models import (
    MultimodalFusionTransformer, 
    CrossModalAttentionModel,
    BilinearFusionModel
)
from utils.logger import get_logger
from utils.metrics import MetricsCalculator
from utils.checkpoint_manager import CheckpointManager
from utils.multimodal_utils import align_modalities, compute_cross_modal_consistency

logger = get_logger(__name__)

class FusionTrainer:
    """Trainer class for multi-modal fusion models"""

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

        # Curriculum learning setup
        self.curriculum_stage = 0
        self.curriculum_stages = self.config['training'].get('curriculum_learning', {}).get('stages', [])

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

        if architecture == 'fusion_transformer':
            model = MultimodalFusionTransformer(
                num_classes=model_config['num_classes'],
                fusion_method=model_config['fusion_method'],
                hidden_dim=model_config['hidden_dim'],
                num_attention_heads=model_config['num_attention_heads'],
                num_layers=model_config['num_layers'],
                dropout_rate=model_config['dropout_rate'],
                modality_encoders=model_config['modality_encoders']
            )
        elif architecture == 'cross_modal_attention':
            model = CrossModalAttentionModel(
                num_classes=model_config['num_classes'],
                **model_config.get('cross_modal_attention', {})
            )
        elif architecture == 'bilinear_fusion':
            model = BilinearFusionModel(
                num_classes=model_config['num_classes'],
                **model_config.get('fusion_strategies', {})
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
        train_dataset = MultimodalDataset(
            data_root=dataset_config['data_root'],
            split='train',
            config=dataset_config
        )

        val_dataset = MultimodalDataset(
            data_root=dataset_config['data_root'],
            split='val',
            config=dataset_config
        )

        test_dataset = MultimodalDataset(
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

        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            **scheduler_config['params']
        )

        return scheduler

    def _build_loss_function(self) -> nn.Module:
        """Build loss function with auxiliary losses"""
        loss_config = self.config['training']['loss']

        # Main loss
        main_criterion = nn.CrossEntropyLoss(
            label_smoothing=loss_config['params'].get('label_smoothing', 0.0)
        )

        return MultimodalCriterion(
            main_criterion=main_criterion,
            auxiliary_losses=loss_config.get('auxiliary_losses', {}),
            modality_weights=self.config['training'].get('modality_weights', {}),
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

    def _update_curriculum_stage(self):
        """Update curriculum learning stage"""
        if not self.curriculum_stages:
            return

        for i, stage in enumerate(self.curriculum_stages):
            stage_epochs = sum(s['epochs'] for s in self.curriculum_stages[:i+1])
            if self.current_epoch < stage_epochs:
                if i != self.curriculum_stage:
                    self.curriculum_stage = i
                    logger.info(f"Moving to curriculum stage {i+1}: {stage}")
                break

    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        running_losses = defaultdict(float)
        running_correct = 0
        running_total = 0

        # Update curriculum stage
        self._update_curriculum_stage()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch_data in enumerate(progress_bar):
            # Unpack multimodal data
            visual_data = batch_data['visual'].to(self.device)
            audio_data = batch_data['audio'].to(self.device)
            temporal_data = batch_data.get('temporal', None)
            if temporal_data is not None:
                temporal_data = temporal_data.to(self.device)
            targets = batch_data['label'].to(self.device)

            # Prepare input dictionary
            inputs = {
                'visual': visual_data,
                'audio': audio_data,
                'temporal': temporal_data
            }

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Calculate loss
            loss_dict = self.criterion(outputs, targets, inputs)
            total_loss = loss_dict['total_loss']

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Statistics
            for loss_name, loss_value in loss_dict.items():
                running_losses[loss_name] += loss_value.item() if hasattr(loss_value, 'item') else loss_value

            if 'main_output' in outputs:
                _, predicted = torch.max(outputs['main_output'].data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)

            running_total += targets.size(0)
            running_correct += (predicted == targets).sum().item()

            # Update progress bar
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Acc': f'{100.*running_correct/running_total:.2f}%',
                    'LR': f'{current_lr:.6f}',
                    'Stage': f'{self.curriculum_stage+1}'
                })

        # Calculate epoch metrics
        epoch_metrics = {}
        for loss_name, loss_sum in running_losses.items():
            epoch_metrics[loss_name] = loss_sum / len(self.train_loader)

        epoch_metrics['accuracy'] = 100. * running_correct / running_total

        return epoch_metrics

    def validate_epoch(self) -> dict:
        """Validate for one epoch"""
        self.model.eval()
        running_losses = defaultdict(float)
        all_predictions = []
        all_targets = []
        modality_contributions = defaultdict(list)

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validation"):
                # Unpack multimodal data
                visual_data = batch_data['visual'].to(self.device)
                audio_data = batch_data['audio'].to(self.device)
                temporal_data = batch_data.get('temporal', None)
                if temporal_data is not None:
                    temporal_data = temporal_data.to(self.device)
                targets = batch_data['label'].to(self.device)

                inputs = {
                    'visual': visual_data,
                    'audio': audio_data,
                    'temporal': temporal_data
                }

                outputs = self.model(inputs)

                # Calculate loss
                loss_dict = self.criterion(outputs, targets, inputs)
                for loss_name, loss_value in loss_dict.items():
                    running_losses[loss_name] += loss_value.item() if hasattr(loss_value, 'item') else loss_value

                # Store predictions and targets for metrics calculation
                if 'main_output' in outputs:
                    main_output = outputs['main_output']
                else:
                    main_output = outputs

                probabilities = torch.softmax(main_output, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                # Store modality contributions if available
                if 'modality_contributions' in outputs:
                    for modality, contribution in outputs['modality_contributions'].items():
                        modality_contributions[modality].append(contribution.mean().cpu().item())

        # Calculate metrics
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        metrics = self.metrics_calculator.calculate_metrics(
            predictions=all_predictions,
            targets=all_targets
        )

        # Add loss metrics
        for loss_name, loss_sum in running_losses.items():
            metrics[loss_name] = loss_sum / len(self.val_loader)

        # Add modality contribution analysis
        if modality_contributions:
            for modality, contributions in modality_contributions.items():
                metrics[f'{modality}_contribution'] = np.mean(contributions)

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
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                          f"Val Loss: {val_metrics['total_loss']:.4f}, "
                          f"Val AUC: {val_metrics.get('auc_roc', 0):.4f}")

                # Log to wandb
                if wandb.run:
                    log_dict = {
                        'epoch': epoch,
                        'curriculum_stage': self.curriculum_stage + 1,
                        **{f'train_{k}': v for k, v in train_metrics.items()},
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    }
                    wandb.log(log_dict)

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
                        is_best=True,
                        additional_info={'curriculum_stage': self.curriculum_stage}
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
                    is_best=False,
                    additional_info={'curriculum_stage': self.curriculum_stage}
                )

        logger.info("Training completed!")

    def test(self):
        """Test the best model with comprehensive evaluation"""
        logger.info("Testing best model...")

        # Load best checkpoint
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])

        self.model.eval()
        all_predictions = []
        all_targets = []
        modality_ablation_results = {}
        attention_maps = []

        # Test with all modalities
        with torch.no_grad():
            for batch_data in tqdm(self.test_loader, desc="Testing"):
                visual_data = batch_data['visual'].to(self.device)
                audio_data = batch_data['audio'].to(self.device)
                temporal_data = batch_data.get('temporal', None)
                if temporal_data is not None:
                    temporal_data = temporal_data.to(self.device)
                targets = batch_data['label'].to(self.device)

                inputs = {
                    'visual': visual_data,
                    'audio': audio_data,
                    'temporal': temporal_data
                }

                outputs = self.model(inputs)

                if 'main_output' in outputs:
                    main_output = outputs['main_output']
                else:
                    main_output = outputs

                probabilities = torch.softmax(main_output, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

                # Store attention maps if available
                if 'attention_maps' in outputs:
                    attention_maps.extend(outputs['attention_maps'])

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

        # Modality ablation study
        if self.config['evaluation'].get('robustness_tests'):
            modality_ablation_results = self._run_modality_ablation()
            test_metrics['modality_ablation'] = modality_ablation_results

        # Save test results
        results_path = os.path.join(self.config['paths']['output_root'], 'test_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)

        return test_metrics

    def _run_modality_ablation(self) -> dict:
        """Run modality ablation study"""
        logger.info("Running modality ablation study...")

        modalities = ['visual', 'audio', 'temporal']
        ablation_results = {}

        for removed_modality in modalities:
            logger.info(f"Testing without {removed_modality} modality...")

            predictions = []
            targets = []

            with torch.no_grad():
                for batch_data in tqdm(self.test_loader, desc=f"Ablation - no {removed_modality}"):
                    visual_data = batch_data['visual'].to(self.device)
                    audio_data = batch_data['audio'].to(self.device)
                    temporal_data = batch_data.get('temporal', None)
                    if temporal_data is not None:
                        temporal_data = temporal_data.to(self.device)
                    batch_targets = batch_data['label'].to(self.device)

                    # Prepare inputs with removed modality
                    inputs = {
                        'visual': visual_data if removed_modality != 'visual' else None,
                        'audio': audio_data if removed_modality != 'audio' else None,
                        'temporal': temporal_data if removed_modality != 'temporal' else None
                    }

                    outputs = self.model(inputs)

                    if 'main_output' in outputs:
                        main_output = outputs['main_output']
                    else:
                        main_output = outputs

                    probabilities = torch.softmax(main_output, dim=1)
                    predictions.append(probabilities.cpu().numpy())
                    targets.append(batch_targets.cpu().numpy())

            # Calculate metrics for this ablation
            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)

            metrics = self.metrics_calculator.calculate_metrics(
                predictions=predictions,
                targets=targets
            )

            ablation_results[f'without_{removed_modality}'] = metrics

        return ablation_results

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1

        # Load curriculum stage if available
        if 'additional_info' in checkpoint and 'curriculum_stage' in checkpoint['additional_info']:
            self.curriculum_stage = checkpoint['additional_info']['curriculum_stage']

        logger.info(f"Resumed from checkpoint at epoch {checkpoint['epoch']}")


class MultimodalCriterion(nn.Module):
    """Composite loss function for multimodal models"""

    def __init__(self, main_criterion, auxiliary_losses, modality_weights, device):
        super().__init__()
        self.main_criterion = main_criterion
        self.auxiliary_losses = auxiliary_losses
        self.modality_weights = modality_weights
        self.device = device

    def forward(self, outputs, targets, inputs=None):
        """Forward pass with main loss and auxiliary losses"""
        # Main loss
        if 'main_output' in outputs:
            main_loss = self.main_criterion(outputs['main_output'], targets)
        else:
            main_loss = self.main_criterion(outputs, targets)

        loss_dict = {'main_loss': main_loss}
        total_loss = main_loss

        # Modality-specific losses
        for modality, weight in self.modality_weights.items():
            if f'{modality}_output' in outputs:
                modality_loss = self.main_criterion(outputs[f'{modality}_output'], targets)
                loss_dict[f'{modality}_loss'] = modality_loss
                total_loss += weight * modality_loss

        # Auxiliary losses
        if self.auxiliary_losses.get('cross_modal_consistency', {}).get('enabled') and inputs:
            consistency_loss = self._cross_modal_consistency_loss(outputs, inputs)
            weight = self.auxiliary_losses['cross_modal_consistency']['weight']
            loss_dict['consistency_loss'] = consistency_loss
            total_loss += weight * consistency_loss

        if self.auxiliary_losses.get('contrastive_loss', {}).get('enabled'):
            contrastive_loss = self._contrastive_loss(outputs, targets)
            weight = self.auxiliary_losses['contrastive_loss']['weight']
            loss_dict['contrastive_loss'] = contrastive_loss
            total_loss += weight * contrastive_loss

        loss_dict['total_loss'] = total_loss
        return loss_dict

    def _cross_modal_consistency_loss(self, outputs, inputs):
        """Calculate cross-modal consistency loss"""
        # Implement cross-modal consistency logic
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def _contrastive_loss(self, outputs, targets):
        """Calculate contrastive loss"""
        # Implement contrastive loss logic
        return torch.tensor(0.0, device=self.device, requires_grad=True)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train multimodal fusion deepfake detection model")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--test_only', action='store_true',
                       help='Only run testing')

    args = parser.parse_args()

    # Create trainer
    trainer = FusionTrainer(
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

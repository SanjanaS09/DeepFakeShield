#!/usr/bin/env python3
"""
Hyperparameter Tuning Script
Multi-Modal Deepfake Detection System - Automated Hyperparameter Optimization

This script provides automated hyperparameter tuning using various optimization
strategies including Grid Search, Random Search, Bayesian Optimization, and more.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import itertools
import random
from sklearn.model_selection import ParameterGrid, ParameterSampler
import optuna
from optuna.samplers import TPESampler
import wandb

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger
from utils.metrics import MetricsCalculator
from training.train_image_model import ImageTrainer
from training.train_video_model import VideoTrainer
from training.train_audio_model import AudioTrainer
from training.train_fusion_model import FusionTrainer

logger = get_logger(__name__)

class HyperparameterTuner:
    """Automated hyperparameter tuning framework"""

    def __init__(self, config_path: str, output_dir: str = "tuning_results"):
        """
        Initialize hyperparameter tuner

        Args:
            config_path: Path to tuning configuration
            output_dir: Directory for tuning outputs
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.modality = self.config['modality']
        self.tuning_method = self.config['tuning_method']
        self.search_space = self.config['search_space']
        self.objective_metric = self.config['objective_metric']
        self.optimization_direction = self.config.get('optimization_direction', 'maximize')

        # Results storage
        self.trial_results = []
        self.best_params = None
        self.best_score = float('-inf') if self.optimization_direction == 'maximize' else float('inf')

        # Setup experiment tracking
        self._setup_experiment_tracking()

    def _load_config(self, config_path: str) -> dict:
        """Load tuning configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_experiment_tracking(self):
        """Setup experiment tracking with Wandb"""
        if self.config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=self.config['wandb']['project'],
                name=f"hparam_tuning_{self.modality}_{self.tuning_method}",
                config=self.config,
                tags=['hyperparameter_tuning', self.modality, self.tuning_method]
            )

    def tune_hyperparameters(self) -> Dict[str, Any]:
        """
        Run hyperparameter tuning based on specified method

        Returns:
            Dictionary containing best parameters and results
        """
        logger.info(f"Starting hyperparameter tuning for {self.modality} modality using {self.tuning_method}")

        if self.tuning_method == 'grid_search':
            return self._grid_search()
        elif self.tuning_method == 'random_search':
            return self._random_search()
        elif self.tuning_method == 'bayesian_optimization':
            return self._bayesian_optimization()
        elif self.tuning_method == 'optuna':
            return self._optuna_optimization()
        else:
            raise ValueError(f"Unknown tuning method: {self.tuning_method}")

    def _grid_search(self) -> Dict[str, Any]:
        """Grid search hyperparameter tuning"""
        logger.info("Running Grid Search...")

        # Generate parameter grid
        param_grid = ParameterGrid(self.search_space)
        total_trials = len(param_grid)

        logger.info(f"Total parameter combinations: {total_trials}")

        for trial_idx, params in enumerate(param_grid):
            logger.info(f"Trial {trial_idx + 1}/{total_trials}: {params}")

            # Evaluate parameter combination
            score = self._evaluate_parameters(params, trial_idx)

            # Update best parameters
            self._update_best_parameters(params, score)

            # Log progress
            self._log_trial_result(trial_idx, params, score)

        return self._get_tuning_results()

    def _random_search(self) -> Dict[str, Any]:
        """Random search hyperparameter tuning"""
        logger.info("Running Random Search...")

        n_trials = self.config.get('n_trials', 50)
        param_sampler = ParameterSampler(self.search_space, n_iter=n_trials, random_state=42)

        for trial_idx, params in enumerate(param_sampler):
            logger.info(f"Trial {trial_idx + 1}/{n_trials}: {params}")

            # Evaluate parameter combination
            score = self._evaluate_parameters(params, trial_idx)

            # Update best parameters
            self._update_best_parameters(params, score)

            # Log progress
            self._log_trial_result(trial_idx, params, score)

        return self._get_tuning_results()

    def _bayesian_optimization(self) -> Dict[str, Any]:
        """Bayesian optimization hyperparameter tuning"""
        logger.info("Running Bayesian Optimization...")

        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            logger.error("scikit-optimize not installed. Please install with: pip install scikit-optimize")
            raise

        # Convert search space to skopt format
        dimensions = []
        param_names = []

        for param_name, param_config in self.search_space.items():
            param_names.append(param_name)

            if param_config['type'] == 'float':
                dimensions.append(Real(param_config['min'], param_config['max'], name=param_name))
            elif param_config['type'] == 'int':
                dimensions.append(Integer(param_config['min'], param_config['max'], name=param_name))
            elif param_config['type'] == 'categorical':
                dimensions.append(Categorical(param_config['choices'], name=param_name))

        @use_named_args(dimensions)
        def objective(**params):
            """Objective function for optimization"""
            score = self._evaluate_parameters(params, len(self.trial_results))

            # Convert to minimization problem if necessary
            if self.optimization_direction == 'maximize':
                return -score
            else:
                return score

        # Run optimization
        n_calls = self.config.get('n_trials', 50)
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )

        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun if self.optimization_direction == 'maximize' else result.fun

        self.best_params = best_params
        self.best_score = best_score

        return self._get_tuning_results()

    def _optuna_optimization(self) -> Dict[str, Any]:
        """Optuna-based hyperparameter optimization"""
        logger.info("Running Optuna Optimization...")

        def objective(trial):
            """Objective function for Optuna"""
            params = {}

            for param_name, param_config in self.search_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['min'], param_config['max']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
                elif param_config['type'] == 'log_uniform':
                    params[param_name] = trial.suggest_loguniform(
                        param_name, param_config['min'], param_config['max']
                    )

            # Evaluate parameters
            score = self._evaluate_parameters(params, trial.number)

            return score

        # Create study
        direction = 'maximize' if self.optimization_direction == 'maximize' else 'minimize'
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=42),
            study_name=f'{self.modality}_hyperparameter_tuning'
        )

        # Optimize
        n_trials = self.config.get('n_trials', 50)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Extract best results
        self.best_params = study.best_params
        self.best_score = study.best_value

        # Save study results
        study_file = self.output_dir / 'optuna_study.pkl'
        import joblib
        joblib.dump(study, study_file)

        return self._get_tuning_results()

    def _evaluate_parameters(self, params: Dict[str, Any], trial_idx: int) -> float:
        """
        Evaluate a set of hyperparameters

        Args:
            params: Hyperparameters to evaluate
            trial_idx: Trial index for logging

        Returns:
            Evaluation score
        """
        try:
            # Create modified configuration
            trial_config = self._create_trial_config(params)

            # Create trainer based on modality
            trainer = self._create_trainer(trial_config, trial_idx)

            # Run training (with early stopping for efficiency)
            trainer.train()

            # Get validation score
            if hasattr(trainer, 'best_val_metrics'):
                score = trainer.best_val_metrics.get(self.objective_metric, 0)
            else:
                # Run validation if not available
                val_metrics = trainer.validate_epoch()
                score = val_metrics.get(self.objective_metric, 0)

            # Store trial result
            trial_result = {
                'trial_idx': trial_idx,
                'params': params.copy(),
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            self.trial_results.append(trial_result)

            logger.info(f"Trial {trial_idx} completed. Score: {score:.4f}")

            return score

        except Exception as e:
            logger.error(f"Trial {trial_idx} failed: {e}")
            # Return worst possible score
            return float('-inf') if self.optimization_direction == 'maximize' else float('inf')

    def _create_trial_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration for a specific trial"""
        base_config = self.config['base_config'].copy()

        # Update configuration with trial parameters
        for param_path, param_value in params.items():
            self._set_nested_config_value(base_config, param_path, param_value)

        # Modify for efficient tuning
        base_config['training']['num_epochs'] = min(
            base_config['training']['num_epochs'],
            self.config.get('max_epochs_per_trial', 20)
        )

        # Enable early stopping for efficiency
        if 'regularization' not in base_config['training']:
            base_config['training']['regularization'] = {}
        if 'early_stopping' not in base_config['training']['regularization']:
            base_config['training']['regularization']['early_stopping'] = {}

        base_config['training']['regularization']['early_stopping']['patience'] = min(
            base_config['training']['regularization']['early_stopping'].get('patience', 10),
            self.config.get('early_stopping_patience', 5)
        )

        return base_config

    def _set_nested_config_value(self, config: Dict, path: str, value: Any):
        """Set value in nested configuration dictionary"""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _create_trainer(self, config: Dict[str, Any], trial_idx: int):
        """Create trainer based on modality"""
        # Save trial config
        trial_config_path = self.output_dir / f'trial_{trial_idx}_config.yaml'
        with open(trial_config_path, 'w') as f:
            yaml.dump(config, f)

        # Create trainer
        if self.modality == 'image':
            return ImageTrainer(config_path=str(trial_config_path))
        elif self.modality == 'video':
            return VideoTrainer(config_path=str(trial_config_path))
        elif self.modality == 'audio':
            return AudioTrainer(config_path=str(trial_config_path))
        elif self.modality == 'fusion':
            return FusionTrainer(config_path=str(trial_config_path))
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    def _update_best_parameters(self, params: Dict[str, Any], score: float):
        """Update best parameters if current score is better"""
        is_better = (
            (self.optimization_direction == 'maximize' and score > self.best_score) or
            (self.optimization_direction == 'minimize' and score < self.best_score)
        )

        if is_better:
            self.best_params = params.copy()
            self.best_score = score
            logger.info(f"New best score: {score:.4f} with params: {params}")

    def _log_trial_result(self, trial_idx: int, params: Dict[str, Any], score: float):
        """Log trial result to various tracking systems"""
        # Log to Wandb if enabled
        if wandb.run:
            log_dict = {
                'trial': trial_idx,
                'score': score,
                **{f'param_{k}': v for k, v in params.items()}
            }
            wandb.log(log_dict)

        # Save intermediate results
        results_file = self.output_dir / 'intermediate_results.json'
        with open(results_file, 'w') as f:
            json.dump({
                'trial_results': self.trial_results,
                'best_params': self.best_params,
                'best_score': self.best_score
            }, f, indent=2)

    def _get_tuning_results(self) -> Dict[str, Any]:
        """Get final tuning results"""
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_trials': len(self.trial_results),
            'objective_metric': self.objective_metric,
            'optimization_direction': self.optimization_direction,
            'tuning_method': self.tuning_method,
            'modality': self.modality,
            'all_trials': self.trial_results,
            'tuning_config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        # Save final results
        final_results_file = self.output_dir / 'final_results.json'
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Create best configuration file
        best_config = self._create_trial_config(self.best_params)
        best_config_file = self.output_dir / 'best_config.yaml'
        with open(best_config_file, 'w') as f:
            yaml.dump(best_config, f)

        logger.info(f"Hyperparameter tuning completed!")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Results saved to: {self.output_dir}")

        return results

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze tuning results and generate insights"""
        if not self.trial_results:
            return {}

        analysis = {}

        # Parameter importance analysis
        param_importance = self._analyze_parameter_importance()
        analysis['parameter_importance'] = param_importance

        # Convergence analysis
        convergence_data = self._analyze_convergence()
        analysis['convergence'] = convergence_data

        # Performance distribution analysis
        performance_stats = self._analyze_performance_distribution()
        analysis['performance_statistics'] = performance_stats

        # Save analysis results
        analysis_file = self.output_dir / 'analysis_results.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        return analysis

    def _analyze_parameter_importance(self) -> Dict[str, Any]:
        """Analyze importance of different parameters"""
        if len(self.trial_results) < 10:
            return {}

        # Extract parameter values and scores
        param_names = list(self.search_space.keys())
        param_values = {name: [] for name in param_names}
        scores = []

        for trial in self.trial_results:
            scores.append(trial['score'])
            for param_name in param_names:
                param_values[param_name].append(trial['params'].get(param_name))

        # Calculate correlations (simplified analysis)
        importance = {}
        for param_name, values in param_values.items():
            try:
                # Convert to numeric if possible
                numeric_values = []
                numeric_scores = []

                for val, score in zip(values, scores):
                    if isinstance(val, (int, float)):
                        numeric_values.append(val)
                        numeric_scores.append(score)

                if len(numeric_values) > 5:
                    correlation = np.corrcoef(numeric_values, numeric_scores)[0, 1]
                    importance[param_name] = {
                        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                        'n_samples': len(numeric_values)
                    }
            except Exception:
                importance[param_name] = {'correlation': 0.0, 'n_samples': 0}

        return importance

    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence of the tuning process"""
        scores = [trial['score'] for trial in self.trial_results]

        # Calculate running best scores
        running_best = []
        current_best = float('-inf') if self.optimization_direction == 'maximize' else float('inf')

        for score in scores:
            if self.optimization_direction == 'maximize':
                current_best = max(current_best, score)
            else:
                current_best = min(current_best, score)
            running_best.append(current_best)

        return {
            'scores': scores,
            'running_best': running_best,
            'final_best': self.best_score,
            'improvement_trials': [i for i, (curr, prev) in enumerate(zip(running_best[1:], running_best[:-1])) if curr != prev]
        }

    def _analyze_performance_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of performance scores"""
        scores = [trial['score'] for trial in self.trial_results]

        return {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
            'count': len(scores)
        }


def create_tuning_config_template(modality: str, output_path: str):
    """Create a template configuration for hyperparameter tuning"""

    if modality == 'image':
        search_space = {
            'training.learning_rate': {
                'type': 'log_uniform',
                'min': 1e-5,
                'max': 1e-2
            },
            'training.weight_decay': {
                'type': 'log_uniform', 
                'min': 1e-6,
                'max': 1e-3
            },
            'training.batch_size': {
                'type': 'categorical',
                'choices': [16, 32, 64]
            },
            'model.dropout_rate': {
                'type': 'float',
                'min': 0.1,
                'max': 0.7
            },
            'training.optimizer.params.betas': {
                'type': 'categorical',
                'choices': [[0.9, 0.999], [0.95, 0.999], [0.9, 0.99]]
            }
        }
    elif modality == 'video':
        search_space = {
            'training.learning_rate': {
                'type': 'log_uniform',
                'min': 1e-5,
                'max': 5e-3
            },
            'training.batch_size': {
                'type': 'categorical',
                'choices': [4, 8, 16]
            },
            'model.dropout_rate': {
                'type': 'float',
                'min': 0.1,
                'max': 0.6
            },
            'training.gradient_clip_value': {
                'type': 'float',
                'min': 0.5,
                'max': 2.0
            }
        }
    elif modality == 'audio':
        search_space = {
            'training.learning_rate': {
                'type': 'log_uniform',
                'min': 1e-5,
                'max': 1e-2
            },
            'training.batch_size': {
                'type': 'categorical',
                'choices': [8, 16, 32]
            },
            'model.channels': {
                'type': 'categorical',
                'choices': [
                    [512, 512, 512, 512, 1536],
                    [1024, 1024, 1024, 1024, 3072],
                    [1536, 1536, 1536, 1536, 4608]
                ]
            }
        }
    else:  # fusion
        search_space = {
            'training.learning_rate': {
                'type': 'log_uniform',
                'min': 1e-5,
                'max': 5e-3
            },
            'model.hidden_dim': {
                'type': 'categorical',
                'choices': [256, 512, 768, 1024]
            },
            'model.num_attention_heads': {
                'type': 'categorical',
                'choices': [4, 8, 12, 16]
            },
            'model.num_layers': {
                'type': 'int',
                'min': 2,
                'max': 6
            }
        }

    config = {
        'modality': modality,
        'tuning_method': 'optuna',  # optuna, bayesian_optimization, random_search, grid_search
        'objective_metric': 'auc_roc',
        'optimization_direction': 'maximize',
        'n_trials': 100,
        'max_epochs_per_trial': 15,
        'early_stopping_patience': 3,
        'search_space': search_space,
        'base_config': f'training/configs/{modality}_config.yaml',
        'wandb': {
            'enabled': True,
            'project': f'deepfake_detection_{modality}_tuning'
        }
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f, indent=2)

    logger.info(f"Created tuning config template: {output_path}")


def main():
    """Main hyperparameter tuning function"""
    parser = argparse.ArgumentParser(description="Automated hyperparameter tuning")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to tuning configuration file')
    parser.add_argument('--output_dir', type=str, default='tuning_results',
                       help='Output directory for tuning results')
    parser.add_argument('--create_template', type=str, choices=['image', 'video', 'audio', 'fusion'],
                       help='Create a template configuration for specified modality')
    parser.add_argument('--template_output', type=str, default='tuning_config_template.yaml',
                       help='Output path for template configuration')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only run analysis on existing results')

    args = parser.parse_args()

    # Create template if requested
    if args.create_template:
        create_tuning_config_template(args.create_template, args.template_output)
        return

    # Create tuner
    tuner = HyperparameterTuner(
        config_path=args.config,
        output_dir=args.output_dir
    )

    if args.analyze_only:
        # Only run analysis
        analysis = tuner.analyze_results()
        logger.info("Analysis completed")
    else:
        # Run hyperparameter tuning
        results = tuner.tune_hyperparameters()

        # Run analysis
        analysis = tuner.analyze_results()

        logger.info("Hyperparameter tuning and analysis completed!")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best score: {results['best_score']:.4f}")


if __name__ == '__main__':
    main()

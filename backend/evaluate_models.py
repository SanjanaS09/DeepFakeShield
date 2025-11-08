#!/usr/bin/env python3
"""
Model Evaluation Script
Multi-Modal Deepfake Detection System - Comprehensive Model Evaluation

This script provides comprehensive evaluation of trained models across all modalities
with detailed performance analysis, visualizations, and comparison reports.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.logger import get_logger
# from utils.metrics import MetricsCalculator, compute_eer  # ← Comment out
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
    plot_feature_importance, create_performance_dashboard
)
from models.model_loader import load_trained_model
from data_loaders.evaluation_dataloader import EvaluationDataLoader

logger = get_logger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation framework"""

    def __init__(self, config_path: str, output_dir: str = "evaluation_results"):
        """
        Initialize evaluator

        Args:
            config_path: Path to evaluation configuration
            output_dir: Directory for evaluation outputs
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._setup_device()
        self.metrics_calculator = MetricsCalculator()

        # Results storage
        self.model_results = {}
        self.comparison_results = {}

    def _load_config(self, config_path: str) -> dict:
        """Load evaluation configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_device(self) -> torch.device:
        """Setup evaluation device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device

    def evaluate_single_model(self, model_config: dict) -> dict:
        """
        Evaluate a single model

        Args:
            model_config: Model configuration dictionary

        Returns:
            Dictionary containing evaluation results
        """
        model_name = model_config['name']
        logger.info(f"Evaluating model: {model_name}")

        # Load model
        model = load_trained_model(
            model_path=model_config['model_path'],
            config_path=model_config.get('config_path'),
            device=self.device
        )
        model.eval()

        # Load data
        data_loader = EvaluationDataLoader(
            data_config=model_config['data_config'],
            modality=model_config['modality']
        )
        test_loader = data_loader.get_test_loader()

        # Run evaluation
        results = self._evaluate_model_on_dataset(model, test_loader, model_name)

        # Additional analyses
        if model_config.get('run_robustness_tests', False):
            results['robustness'] = self._run_robustness_tests(model, data_loader, model_name)

        if model_config.get('analyze_failures', False):
            results['failure_analysis'] = self._analyze_failure_cases(model, test_loader, model_name)

        if model_config.get('compute_feature_importance', False):
            results['feature_importance'] = self._compute_feature_importance(model, test_loader, model_name)

        self.model_results[model_name] = results

        # Save individual model results
        self._save_model_results(model_name, results)

        return results

    def _evaluate_model_on_dataset(self, model: nn.Module, data_loader, model_name: str) -> dict:
        """Evaluate model on dataset"""
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_features = []
        inference_times = []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # Handle different data formats
                if isinstance(batch_data, dict):
                    # Multi-modal data
                    inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                             for k, v in batch_data.items() if k != 'label'}
                    targets = batch_data['label'].to(self.device)
                else:
                    # Single modality data
                    inputs, targets = batch_data
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                outputs = model(inputs)
                end_time.record()

                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                inference_times.append(inference_time)

                # Extract predictions
                if isinstance(outputs, dict):
                    logits = outputs.get('main_output', outputs.get('logits', list(outputs.values())[0]))
                else:
                    logits = outputs

                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Extract features if available
                if isinstance(outputs, dict) and 'features' in outputs:
                    all_features.extend(outputs['features'].cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            predictions=all_predictions,
            probabilities=all_probabilities,
            targets=all_targets
        )

        # Add timing metrics
        metrics['inference_time'] = {
            'mean_ms': np.mean(inference_times),
            'std_ms': np.std(inference_times),
            'min_ms': np.min(inference_times),
            'max_ms': np.max(inference_times),
            'median_ms': np.median(inference_times)
        }

        # Store raw results for further analysis
        results = {
            'metrics': metrics,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'features': np.array(all_features) if all_features else None,
            'model_name': model_name
        }

        return results

    def _calculate_comprehensive_metrics(self, predictions, probabilities, targets) -> dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}

        # Basic classification metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='binary')

        metrics.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

        # ROC and PR metrics
        if probabilities.shape[1] == 2:  # Binary classification
            pos_probs = probabilities[:, 1]

            try:
                auc_roc = roc_auc_score(targets, pos_probs)
                auc_pr = average_precision_score(targets, pos_probs)
                eer = compute_eer(targets, pos_probs)

                metrics.update({
                    'auc_roc': auc_roc,
                    'auc_pr': auc_pr,
                    'equal_error_rate': eer
                })

                # Calculate curves for plotting
                fpr, tpr, roc_thresholds = roc_curve(targets, pos_probs)
                precision_curve, recall_curve, pr_thresholds = precision_recall_curve(targets, pos_probs)

                metrics['curves'] = {
                    'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
                    'pr': {'precision': precision_curve, 'recall': recall_curve, 'thresholds': pr_thresholds}
                }
            except Exception as e:
                logger.warning(f"Could not calculate ROC/PR metrics: {e}")

        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': ['Real', 'Deepfake']
        }

        # Class-wise metrics
        class_report = classification_report(targets, predictions, output_dict=True)
        metrics['class_wise'] = class_report

        # Confidence statistics
        confidence_stats = self._analyze_confidence_distribution(probabilities, targets)
        metrics['confidence_analysis'] = confidence_stats

        return metrics

    def _analyze_confidence_distribution(self, probabilities, targets) -> dict:
        """Analyze confidence distribution"""
        max_probs = np.max(probabilities, axis=1)

        # Overall confidence statistics
        stats = {
            'mean_confidence': float(np.mean(max_probs)),
            'std_confidence': float(np.std(max_probs)),
            'min_confidence': float(np.min(max_probs)),
            'max_confidence': float(np.max(max_probs))
        }

        # Confidence by class
        for class_idx in np.unique(targets):
            class_mask = targets == class_idx
            class_confidences = max_probs[class_mask]

            stats[f'class_{class_idx}_confidence'] = {
                'mean': float(np.mean(class_confidences)),
                'std': float(np.std(class_confidences)),
                'count': int(np.sum(class_mask))
            }

        # Calibration analysis
        stats['calibration'] = self._calculate_calibration_metrics(probabilities, targets)

        return stats

    def _calculate_calibration_metrics(self, probabilities, targets) -> dict:
        """Calculate model calibration metrics"""
        from sklearn.calibration import calibration_curve

        try:
            if probabilities.shape[1] == 2:
                pos_probs = probabilities[:, 1]
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    targets, pos_probs, n_bins=10
                )

                # Expected Calibration Error (ECE)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]

                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (pos_probs > bin_lower) & (pos_probs <= bin_upper)
                    prop_in_bin = in_bin.mean()

                    if prop_in_bin > 0:
                        accuracy_in_bin = targets[in_bin].mean()
                        avg_confidence_in_bin = pos_probs[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                return {
                    'expected_calibration_error': float(ece),
                    'calibration_curve': {
                        'fraction_of_positives': fraction_of_positives.tolist(),
                        'mean_predicted_value': mean_predicted_value.tolist()
                    }
                }
        except Exception as e:
            logger.warning(f"Could not calculate calibration metrics: {e}")

        return {}

    def _run_robustness_tests(self, model: nn.Module, data_loader, model_name: str) -> dict:
        """Run robustness tests on the model"""
        logger.info(f"Running robustness tests for {model_name}")

        robustness_results = {}

        # Test with different confidence thresholds
        threshold_results = self._test_confidence_thresholds(model, data_loader)
        robustness_results['threshold_analysis'] = threshold_results

        # Test with corrupted inputs (if applicable)
        if hasattr(data_loader, 'get_corrupted_loader'):
            corruption_results = self._test_input_corruptions(model, data_loader)
            robustness_results['corruption_analysis'] = corruption_results

        # Test with adversarial examples (placeholder)
        # adversarial_results = self._test_adversarial_robustness(model, data_loader)
        # robustness_results['adversarial_analysis'] = adversarial_results

        return robustness_results

    def _test_confidence_thresholds(self, model: nn.Module, data_loader) -> dict:
        """Test model performance at different confidence thresholds"""
        thresholds = np.linspace(0.5, 0.95, 10)
        threshold_results = []

        # Get base predictions
        base_results = self._evaluate_model_on_dataset(model, data_loader.get_test_loader(), "threshold_test")
        probabilities = base_results['probabilities']
        predictions = base_results['predictions']
        targets = base_results['targets']

        for threshold in thresholds:
            max_probs = np.max(probabilities, axis=1)
            confident_mask = max_probs >= threshold

            if np.sum(confident_mask) > 0:
                confident_predictions = predictions[confident_mask]
                confident_targets = targets[confident_mask]

                accuracy = accuracy_score(confident_targets, confident_predictions)
                coverage = np.mean(confident_mask)

                threshold_results.append({
                    'threshold': float(threshold),
                    'accuracy': float(accuracy),
                    'coverage': float(coverage),
                    'num_samples': int(np.sum(confident_mask))
                })

        return threshold_results

    def _analyze_failure_cases(self, model: nn.Module, data_loader, model_name: str) -> dict:
        """Analyze failure cases to understand model weaknesses"""
        logger.info(f"Analyzing failure cases for {model_name}")

        # Get predictions and identify failures
        results = self._evaluate_model_on_dataset(model, data_loader, f"{model_name}_failure_analysis")

        predictions = results['predictions']
        targets = results['targets']
        probabilities = results['probabilities']

        # Identify different types of failures
        false_positives = (predictions == 1) & (targets == 0)  # Real classified as Deepfake
        false_negatives = (predictions == 0) & (targets == 1)  # Deepfake classified as Real

        analysis = {
            'false_positives': {
                'count': int(np.sum(false_positives)),
                'percentage': float(np.mean(false_positives) * 100),
                'confidence_stats': {
                    'mean': float(np.mean(np.max(probabilities[false_positives], axis=1))) if np.any(false_positives) else 0,
                    'std': float(np.std(np.max(probabilities[false_positives], axis=1))) if np.any(false_positives) else 0
                }
            },
            'false_negatives': {
                'count': int(np.sum(false_negatives)),
                'percentage': float(np.mean(false_negatives) * 100),
                'confidence_stats': {
                    'mean': float(np.mean(np.max(probabilities[false_negatives], axis=1))) if np.any(false_negatives) else 0,
                    'std': float(np.std(np.max(probabilities[false_negatives], axis=1))) if np.any(false_negatives) else 0
                }
            }
        }

        # Confidence distribution analysis for failures
        if np.any(false_positives) or np.any(false_negatives):
            all_failures = false_positives | false_negatives
            failure_confidences = np.max(probabilities[all_failures], axis=1)
            correct_confidences = np.max(probabilities[~all_failures], axis=1)

            analysis['confidence_comparison'] = {
                'failure_mean_confidence': float(np.mean(failure_confidences)),
                'correct_mean_confidence': float(np.mean(correct_confidences)),
                'confidence_difference': float(np.mean(correct_confidences) - np.mean(failure_confidences))
            }

        return analysis

    def _compute_feature_importance(self, model: nn.Module, data_loader, model_name: str) -> dict:
        """Compute feature importance using various methods"""
        logger.info(f"Computing feature importance for {model_name}")

        # This is a placeholder - implement specific feature importance methods
        # based on your model architectures

        importance_results = {
            'method': 'gradient_based',
            'importance_scores': [],
            'feature_names': []
        }

        return importance_results

    def compare_models(self, model_configs: list) -> dict:
        """Compare multiple models"""
        logger.info("Running model comparison...")

        # Evaluate all models if not already done
        for config in model_configs:
            model_name = config['name']
            if model_name not in self.model_results:
                self.evaluate_single_model(config)

        # Create comparison analysis
        comparison = self._create_model_comparison()

        # Statistical significance testing
        significance_results = self._test_statistical_significance()
        comparison['statistical_tests'] = significance_results

        # Save comparison results
        self._save_comparison_results(comparison)

        return comparison

    def _create_model_comparison(self) -> dict:
        """Create comprehensive model comparison"""
        comparison_data = []

        for model_name, results in self.model_results.items():
            metrics = results['metrics']

            model_summary = {
                'model_name': model_name,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'auc_roc': metrics.get('auc_roc', 0),
                'auc_pr': metrics.get('auc_pr', 0),
                'equal_error_rate': metrics.get('equal_error_rate', 0),
                'mean_inference_time': metrics.get('inference_time', {}).get('mean_ms', 0)
            }

            comparison_data.append(model_summary)

        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_data)

        # Find best performing models for each metric
        best_models = {}
        for metric in ['accuracy', 'auc_roc', 'f1_score']:
            if metric in df_comparison.columns:
                best_idx = df_comparison[metric].idxmax()
                best_models[metric] = {
                    'model': df_comparison.loc[best_idx, 'model_name'],
                    'value': df_comparison.loc[best_idx, metric]
                }

        # Calculate performance rankings
        ranking_metrics = ['accuracy', 'auc_roc', 'f1_score']
        available_metrics = [m for m in ranking_metrics if m in df_comparison.columns]

        if available_metrics:
            # Rank models (higher is better for these metrics)
            rankings = {}
            for metric in available_metrics:
                ranked = df_comparison.nlargest(len(df_comparison), metric)
                rankings[metric] = ranked[['model_name', metric]].to_dict('records')
        else:
            rankings = {}

        comparison = {
            'summary_table': df_comparison.to_dict('records'),
            'best_models': best_models,
            'rankings': rankings,
            'comparison_timestamp': datetime.now().isoformat()
        }

        return comparison

    def _test_statistical_significance(self) -> dict:
        """Test statistical significance between model performances"""
        # Placeholder for statistical significance testing
        # Implement McNemar's test, bootstrap confidence intervals, etc.

        return {
            'method': 'mcnemar_test',
            'results': 'Not implemented - placeholder'
        }

    def generate_visualizations(self):
        """Generate comprehensive visualization suite"""
        logger.info("Generating visualizations...")

        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        # Performance comparison charts
        self._create_performance_comparison_charts(vis_dir)

        # Individual model visualizations
        for model_name, results in self.model_results.items():
            model_vis_dir = vis_dir / model_name
            model_vis_dir.mkdir(exist_ok=True)

            self._create_model_visualizations(results, model_vis_dir)

        # Create interactive dashboard
        self._create_interactive_dashboard(vis_dir)

    def _create_performance_comparison_charts(self, output_dir: Path):
        """Create performance comparison charts"""
        if len(self.model_results) < 2:
            return

        # Extract metrics for comparison
        models = []
        metrics_data = {}

        for model_name, results in self.model_results.items():
            models.append(model_name)
            metrics = results['metrics']

            for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(metrics.get(metric_name, 0))

        # Create comparison bar chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        axes = axes.flatten()

        for i, (metric_name, values) in enumerate(metrics_data.items()):
            if i < len(axes):
                ax = axes[i]
                bars = ax.bar(models, values)
                ax.set_title(f'{metric_name.replace("_", " ").title()}')
                ax.set_ylabel(metric_name)
                ax.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')

        # Remove unused subplots
        for i in range(len(metrics_data), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _create_model_visualizations(self, results: dict, output_dir: Path):
        """Create visualizations for a single model"""
        metrics = results['metrics']

        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix']['matrix'])
            labels = metrics['confusion_matrix']['labels']

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.title(f'Confusion Matrix - {results["model_name"]}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

        # ROC Curve
        if 'curves' in metrics and 'roc' in metrics['curves']:
            roc_data = metrics['curves']['roc']

            plt.figure(figsize=(8, 6))
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f'ROC Curve (AUC = {metrics.get("auc_roc", 0):.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {results["model_name"]}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Precision-Recall Curve
        if 'curves' in metrics and 'pr' in metrics['curves']:
            pr_data = metrics['curves']['pr']

            plt.figure(figsize=(8, 6))
            plt.plot(pr_data['recall'], pr_data['precision'],
                    label=f'PR Curve (AP = {metrics.get("auc_pr", 0):.3f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {results["model_name"]}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _create_interactive_dashboard(self, output_dir: Path):
        """Create interactive Plotly dashboard"""
        # Create comprehensive interactive dashboard
        # This is a simplified version - expand based on requirements

        dashboard_data = []
        for model_name, results in self.model_results.items():
            metrics = results['metrics']
            dashboard_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'AUC-ROC': metrics.get('auc_roc', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'Inference Time (ms)': metrics.get('inference_time', {}).get('mean_ms', 0)
            })

        df = pd.DataFrame(dashboard_data)

        # Create interactive scatter plot
        fig = px.scatter(df, x='Accuracy', y='AUC-ROC', 
                        size='Inference Time (ms)', hover_name='Model',
                        title='Model Performance Overview')

        fig.write_html(output_dir / 'interactive_dashboard.html')

    def _save_model_results(self, model_name: str, results: dict):
        """Save individual model results"""
        # Create model-specific directory
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)

        # Save metrics
        metrics_file = model_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = self._make_json_serializable(results['metrics'])
            json.dump(serializable_metrics, f, indent=2)

        # Save detailed results
        results_file = model_dir / 'detailed_results.npz'
        np.savez_compressed(
            results_file,
            predictions=results['predictions'],
            probabilities=results['probabilities'],
            targets=results['targets'],
            features=results['features'] if results['features'] is not None else np.array([])
        )

        logger.info(f"Saved results for {model_name} to {model_dir}")

    def _save_comparison_results(self, comparison: dict):
        """Save model comparison results"""
        comparison_file = self.output_dir / 'model_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        # Save as CSV for easy viewing
        if 'summary_table' in comparison:
            df = pd.DataFrame(comparison['summary_table'])
            df.to_csv(self.output_dir / 'model_comparison.csv', index=False)

        logger.info(f"Saved comparison results to {self.output_dir}")

    def _make_json_serializable(self, obj):
        """Make object JSON serializable by converting numpy types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        report_lines = []
        report_lines.append("# Model Evaluation Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Executive Summary
        report_lines.append("## Executive Summary")
        if len(self.model_results) > 0:
            best_model = max(self.model_results.items(), 
                           key=lambda x: x[1]['metrics'].get('auc_roc', 0))
            best_name, best_results = best_model
            best_auc = best_results['metrics'].get('auc_roc', 0)

            report_lines.append(f"- Best performing model: {best_name} (AUC-ROC: {best_auc:.4f})")
            report_lines.append(f"- Total models evaluated: {len(self.model_results)}")
        report_lines.append("")

        # Individual Model Results
        report_lines.append("## Individual Model Results")
        for model_name, results in self.model_results.items():
            metrics = results['metrics']
            report_lines.append(f"### {model_name}")
            report_lines.append(f"- Accuracy: {metrics.get('accuracy', 0):.4f}")
            report_lines.append(f"- Precision: {metrics.get('precision', 0):.4f}")
            report_lines.append(f"- Recall: {metrics.get('recall', 0):.4f}")
            report_lines.append(f"- F1-Score: {metrics.get('f1_score', 0):.4f}")
            report_lines.append(f"- AUC-ROC: {metrics.get('auc_roc', 0):.4f}")
            report_lines.append(f"- Equal Error Rate: {metrics.get('equal_error_rate', 0):.4f}")

            if 'inference_time' in metrics:
                avg_time = metrics['inference_time'].get('mean_ms', 0)
                report_lines.append(f"- Average Inference Time: {avg_time:.2f}ms")
            report_lines.append("")

        # Model Comparison
        if hasattr(self, 'comparison_results') and self.comparison_results:
            report_lines.append("## Model Comparison")
            # Add comparison details here
            report_lines.append("")

        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / 'evaluation_report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)

        logger.info(f"Generated evaluation report: {report_file}")
        return report_content


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate trained deepfake detection models")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to evaluation configuration file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--generate_visualizations', action='store_true',
                       help='Generate visualization suite')
    parser.add_argument('--compare_models', action='store_true',
                       help='Run model comparison analysis')

    args = parser.parse_args()

    # Create evaluator
    evaluator = ModelEvaluator(
        config_path=args.config,
        output_dir=args.output_dir
    )

    # Load evaluation configuration
    with open(args.config, 'r') as f:
        eval_config = yaml.safe_load(f)

    # Evaluate individual models
    for model_config in eval_config['models']:
        evaluator.evaluate_single_model(model_config)

    # Compare models if requested
    if args.compare_models and len(eval_config['models']) > 1:
        evaluator.compare_models(eval_config['models'])

    # Generate visualizations
    if args.generate_visualizations:
        evaluator.generate_visualizations()

    # Generate report
    evaluator.generate_report()

    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()

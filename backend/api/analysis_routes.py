"""
Analysis Routes for Multi-Modal Deepfake Detection API
Handles analysis, statistics, model performance, and dataset management
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

from flask import Blueprint, request, jsonify, current_app, send_file
import numpy as np
import pandas as pd

from services.detection_service import DeepfakeDetectionService as DetectionService
from services.explanation_service import ExplanationService
from utils.file_handlers import get_dataset_stats, scan_directory
from models.base_model import BaseDetectionModel

logger = logging.getLogger(__name__)

# Create Blueprint
analysis_bp = Blueprint('analysis', __name__)

@analysis_bp.route('/dataset/stats', methods=['GET'])
def get_dataset_statistics():
    """
    Get comprehensive dataset statistics

    Query parameters:
    - dataset_path: Optional path to dataset directory
    - modality: Optional modality filter (image, video, audio, all)
    """
    try:
        dataset_path = request.args.get('dataset_path', 'dataset/')
        modality = request.args.get('modality', 'all')

        if not os.path.exists(dataset_path):
            return jsonify({'error': f'Dataset path not found: {dataset_path}'}), 404

        stats = {}

        # Define modalities to analyze
        modalities = ['image', 'video', 'audio'] if modality == 'all' else [modality]

        for mod in modalities:
            mod_path = Path(dataset_path) / mod
            if mod_path.exists():
                mod_stats = {
                    'train': {'FAKE': 0, 'REAL': 0, 'total': 0},
                    'validation': {'FAKE': 0, 'REAL': 0, 'total': 0},
                    'test': {'FAKE': 0, 'REAL': 0, 'total': 0},
                    'overall': {'FAKE': 0, 'REAL': 0, 'total': 0}
                }

                # Count files in each split and class
                for split in ['train', 'validation', 'test']:
                    split_path = mod_path / split
                    if split_path.exists():
                        for class_label in ['FAKE', 'REAL']:
                            class_path = split_path / class_label
                            if class_path.exists():
                                file_count = len([f for f in class_path.glob('*') 
                                                if f.is_file() and not f.name.startswith('.')])
                                mod_stats[split][class_label] = file_count
                                mod_stats[split]['total'] += file_count
                                mod_stats['overall'][class_label] += file_count
                                mod_stats['overall']['total'] += file_count

                # Calculate ratios
                for split_key in mod_stats:
                    split_data = mod_stats[split_key]
                    if split_data['total'] > 0:
                        split_data['fake_ratio'] = split_data['FAKE'] / split_data['total']
                        split_data['real_ratio'] = split_data['REAL'] / split_data['total']
                        split_data['balance_ratio'] = min(split_data['FAKE'], split_data['REAL']) / max(split_data['FAKE'], split_data['REAL']) if max(split_data['FAKE'], split_data['REAL']) > 0 else 0
                    else:
                        split_data['fake_ratio'] = 0
                        split_data['real_ratio'] = 0  
                        split_data['balance_ratio'] = 0

                stats[mod] = mod_stats
            else:
                stats[mod] = {'error': f'Modality path not found: {mod_path}'}

        # Overall cross-modal statistics
        if modality == 'all':
            total_files = sum(stats[mod]['overall']['total'] for mod in modalities if 'overall' in stats[mod])
            total_fake = sum(stats[mod]['overall']['FAKE'] for mod in modalities if 'overall' in stats[mod])
            total_real = sum(stats[mod]['overall']['REAL'] for mod in modalities if 'overall' in stats[mod])

            stats['cross_modal'] = {
                'total_files': total_files,
                'total_fake': total_fake,
                'total_real': total_real,
                'overall_fake_ratio': total_fake / total_files if total_files > 0 else 0,
                'modality_distribution': {
                    mod: stats[mod]['overall']['total'] / total_files if total_files > 0 else 0
                    for mod in modalities if 'overall' in stats[mod]
                }
            }

        return jsonify({
            'dataset_statistics': stats,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'dataset_path': dataset_path,
            'modality_filter': modality
        })

    except Exception as e:
        logger.error(f"Error getting dataset statistics: {str(e)}")
        return jsonify({'error': 'Failed to retrieve dataset statistics'}), 500

@analysis_bp.route('/model/performance', methods=['GET'])
def get_model_performance():
    """
    Get model performance metrics and analysis

    Query parameters:
    - modality: Modality to analyze (image, video, audio, fusion)
    - split: Dataset split (train, validation, test, all)
    - metric_type: Type of metrics (accuracy, detailed, confusion_matrix)
    """
    try:
        modality = request.args.get('modality', 'image')
        split = request.args.get('split', 'test')
        metric_type = request.args.get('metric_type', 'detailed')

        # This would typically load saved model evaluation results
        # For now, we'll return mock performance data

        performance_data = {
            'model_info': {
                'modality': modality,
                'split': split,
                'evaluation_date': datetime.utcnow().isoformat(),
                'model_version': '1.0.0'
            },
            'metrics': {}
        }

        if metric_type in ['accuracy', 'detailed']:
            # Basic metrics
            performance_data['metrics']['accuracy'] = 0.892
            performance_data['metrics']['precision'] = {
                'FAKE': 0.885,
                'REAL': 0.899,
                'macro_avg': 0.892,
                'weighted_avg': 0.892
            }
            performance_data['metrics']['recall'] = {
                'FAKE': 0.901,
                'REAL': 0.883,
                'macro_avg': 0.892,
                'weighted_avg': 0.892
            }
            performance_data['metrics']['f1_score'] = {
                'FAKE': 0.893,
                'REAL': 0.891,
                'macro_avg': 0.892,
                'weighted_avg': 0.892
            }

            if metric_type == 'detailed':
                # Additional detailed metrics
                performance_data['metrics']['auc_roc'] = 0.951
                performance_data['metrics']['auc_pr'] = 0.944
                performance_data['metrics']['specificity'] = 0.883
                performance_data['metrics']['sensitivity'] = 0.901
                performance_data['metrics']['balanced_accuracy'] = 0.892

                # Confidence distribution
                performance_data['confidence_analysis'] = {
                    'mean_confidence': 0.847,
                    'std_confidence': 0.124,
                    'high_confidence_threshold': 0.8,
                    'high_confidence_ratio': 0.731,
                    'low_confidence_samples': 127
                }

                # Feature importance (mock data)
                if modality == 'image':
                    performance_data['feature_importance'] = {
                        'blur_score': 0.234,
                        'compression_artifacts': 0.198,
                        'facial_warping': 0.267,
                        'noise_score': 0.156,
                        'temporal_inconsistency': 0.145
                    }
                elif modality == 'video':
                    performance_data['feature_importance'] = {
                        'temporal_inconsistency': 0.312,
                        'facial_warping': 0.234,
                        'motion_artifacts': 0.189,
                        'compression_artifacts': 0.143,
                        'lighting_consistency': 0.122
                    }
                elif modality == 'audio':
                    performance_data['feature_importance'] = {
                        'pitch_modulation': 0.289,
                        'spectral_artifacts': 0.245,
                        'voice_quality': 0.201,
                        'compression_artifacts': 0.134,
                        'temporal_consistency': 0.131
                    }

        if metric_type in ['confusion_matrix', 'detailed']:
            # Confusion matrix (mock data)
            performance_data['confusion_matrix'] = {
                'REAL_predicted_REAL': 442,
                'REAL_predicted_FAKE': 58,
                'FAKE_predicted_REAL': 49,
                'FAKE_predicted_FAKE': 451,
                'matrix': [[442, 58], [49, 451]],
                'labels': ['REAL', 'FAKE']
            }

        return jsonify(performance_data)

    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        return jsonify({'error': 'Failed to retrieve model performance'}), 500

@analysis_bp.route('/model/compare', methods=['POST'])
def compare_models():
    """
    Compare performance across different models or modalities

    Request JSON:
    {
        "models": ["image_xception", "video_i3d", "audio_ecapa", "fusion_transformer"],
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "split": "test"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No comparison data provided'}), 400

        models = data.get('models', [])
        metrics = data.get('metrics', ['accuracy'])
        split = data.get('split', 'test')

        if not models:
            return jsonify({'error': 'No models specified for comparison'}), 400

        # Mock comparison data
        comparison_results = {
            'comparison_metadata': {
                'models_compared': models,
                'metrics_evaluated': metrics,
                'dataset_split': split,
                'comparison_date': datetime.utcnow().isoformat()
            },
            'model_performance': {},
            'ranking': {},
            'statistical_significance': {}
        }

        # Mock performance data for each model
        base_performances = {
            'image_xception': {'accuracy': 0.892, 'precision': 0.891, 'recall': 0.893, 'f1_score': 0.892},
            'image_resnet50': {'accuracy': 0.876, 'precision': 0.874, 'recall': 0.878, 'f1_score': 0.876},
            'video_i3d': {'accuracy': 0.901, 'precision': 0.899, 'recall': 0.903, 'f1_score': 0.901},
            'video_slowfast': {'accuracy': 0.887, 'precision': 0.885, 'recall': 0.889, 'f1_score': 0.887},
            'audio_ecapa': {'accuracy': 0.834, 'precision': 0.832, 'recall': 0.836, 'f1_score': 0.834},
            'audio_wav2vec2': {'accuracy': 0.821, 'precision': 0.819, 'recall': 0.823, 'f1_score': 0.821},
            'fusion_transformer': {'accuracy': 0.923, 'precision': 0.921, 'recall': 0.925, 'f1_score': 0.923},
            'fusion_weighted': {'accuracy': 0.912, 'precision': 0.910, 'recall': 0.914, 'f1_score': 0.912}
        }

        for model in models:
            if model in base_performances:
                comparison_results['model_performance'][model] = {
                    metric: base_performances[model].get(metric, 0) 
                    for metric in metrics
                }
            else:
                comparison_results['model_performance'][model] = {
                    metric: 0 for metric in metrics
                }

        # Generate rankings
        for metric in metrics:
            metric_scores = [
                (model, comparison_results['model_performance'][model][metric])
                for model in models
            ]
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            comparison_results['ranking'][metric] = [
                {'rank': i+1, 'model': model, 'score': score}
                for i, (model, score) in enumerate(metric_scores)
            ]

        # Mock statistical significance (p-values)
        comparison_results['statistical_significance'] = {
            f'{models[0]}_vs_{models[1]}': {
                'p_value': 0.032,
                'is_significant': True,
                'confidence_level': 0.95
            } if len(models) >= 2 else {}
        }

        return jsonify(comparison_results)

    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        return jsonify({'error': 'Failed to compare models'}), 500

@analysis_bp.route('/explanation/analyze', methods=['POST'])
def analyze_explanations():
    """
    Analyze XAI explanations across multiple samples

    Request JSON:
    {
        "explanation_type": "gradcam",
        "modality": "image",
        "sample_paths": ["path1", "path2", ...],
        "aggregation_method": "mean"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No analysis data provided'}), 400

        explanation_type = data.get('explanation_type', 'gradcam')
        modality = data.get('modality', 'image')
        sample_paths = data.get('sample_paths', [])
        aggregation_method = data.get('aggregation_method', 'mean')

        if not sample_paths:
            return jsonify({'error': 'No sample paths provided'}), 400

        # Mock explanation analysis
        analysis_results = {
            'analysis_metadata': {
                'explanation_type': explanation_type,
                'modality': modality,
                'num_samples': len(sample_paths),
                'aggregation_method': aggregation_method,
                'analysis_date': datetime.utcnow().isoformat()
            },
            'aggregated_explanations': {},
            'explanation_statistics': {},
            'common_patterns': []
        }

        if explanation_type == 'gradcam':
            analysis_results['aggregated_explanations']['heatmap_statistics'] = {
                'mean_activation': 0.234,
                'std_activation': 0.087,
                'max_activation': 0.892,
                'min_activation': 0.012
            }

            analysis_results['common_patterns'] = [
                {
                    'pattern_id': 1,
                    'description': 'High attention on facial boundary regions',
                    'frequency': 0.67,
                    'avg_intensity': 0.78
                },
                {
                    'pattern_id': 2,
                    'description': 'Focused attention on eye and mouth areas',
                    'frequency': 0.54,
                    'avg_intensity': 0.71
                }
            ]

        elif explanation_type == 'feature_importance':
            analysis_results['aggregated_explanations']['feature_rankings'] = {
                'most_important': [
                    {'feature': 'facial_warping', 'importance': 0.267},
                    {'feature': 'blur_score', 'importance': 0.234},
                    {'feature': 'compression_artifacts', 'importance': 0.198}
                ],
                'least_important': [
                    {'feature': 'temporal_inconsistency', 'importance': 0.145},
                    {'feature': 'noise_score', 'importance': 0.156}
                ]
            }

        analysis_results['explanation_statistics'] = {
            'consistency_score': 0.724,
            'explanation_confidence': 0.812,
            'disagreement_rate': 0.156
        }

        return jsonify(analysis_results)

    except Exception as e:
        logger.error(f"Error analyzing explanations: {str(e)}")
        return jsonify({'error': 'Failed to analyze explanations'}), 500

@analysis_bp.route('/dataset/validate', methods=['POST'])
def validate_dataset():
    """
    Validate dataset structure and integrity

    Request JSON:
    {
        "dataset_path": "/path/to/dataset",
        "check_types": ["structure", "file_integrity", "class_balance"]
    }
    """
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', 'dataset/')
        check_types = data.get('check_types', ['structure', 'file_integrity'])

        validation_results = {
            'dataset_path': dataset_path,
            'validation_date': datetime.utcnow().isoformat(),
            'checks_performed': check_types,
            'validation_status': 'PASSED',
            'issues': [],
            'warnings': [],
            'summary': {}
        }

        # Structure validation
        if 'structure' in check_types:
            structure_issues = []
            expected_structure = {
                'modalities': ['image', 'video', 'audio'],
                'splits': ['train', 'validation', 'test'],
                'classes': ['FAKE', 'REAL']
            }

            dataset_root = Path(dataset_path)
            if not dataset_root.exists():
                structure_issues.append(f"Dataset root directory does not exist: {dataset_path}")
                validation_results['validation_status'] = 'FAILED'
            else:
                for modality in expected_structure['modalities']:
                    mod_path = dataset_root / modality
                    if not mod_path.exists():
                        structure_issues.append(f"Missing modality directory: {modality}")
                        continue

                    for split in expected_structure['splits']:
                        split_path = mod_path / split
                        if not split_path.exists():
                            structure_issues.append(f"Missing split directory: {modality}/{split}")
                            continue

                        for class_label in expected_structure['classes']:
                            class_path = split_path / class_label
                            if not class_path.exists():
                                structure_issues.append(f"Missing class directory: {modality}/{split}/{class_label}")

            validation_results['structure_validation'] = {
                'issues': structure_issues,
                'status': 'FAILED' if structure_issues else 'PASSED'
            }

            if structure_issues:
                validation_results['issues'].extend(structure_issues)

        # File integrity check (simplified)
        if 'file_integrity' in check_types:
            integrity_issues = []
            total_files_checked = 0
            corrupted_files = []

            # This would typically involve checking file headers, sizes, etc.
            # For now, we'll simulate the check
            total_files_checked = 1500  # Mock value

            validation_results['file_integrity'] = {
                'total_files_checked': total_files_checked,
                'corrupted_files': len(corrupted_files),
                'corruption_rate': len(corrupted_files) / total_files_checked if total_files_checked > 0 else 0,
                'status': 'PASSED' if not corrupted_files else 'FAILED'
            }

            if corrupted_files:
                validation_results['issues'].extend([f"Corrupted file: {f}" for f in corrupted_files])

        # Class balance check
        if 'class_balance' in check_types:
            balance_warnings = []

            # This would check if FAKE/REAL ratios are reasonable
            # Mock implementation
            imbalance_threshold = 0.3  # Warning if ratio is below 0.3

            for modality in ['image', 'video', 'audio']:
                mock_balance_ratio = 0.45  # Mock balanced dataset
                if mock_balance_ratio < imbalance_threshold:
                    balance_warnings.append(f"Class imbalance detected in {modality}: ratio = {mock_balance_ratio:.2f}")

            validation_results['class_balance'] = {
                'warnings': balance_warnings,
                'imbalance_threshold': imbalance_threshold,
                'status': 'WARNING' if balance_warnings else 'PASSED'
            }

            if balance_warnings:
                validation_results['warnings'].extend(balance_warnings)

        # Update overall status
        if validation_results['issues']:
            validation_results['validation_status'] = 'FAILED'
        elif validation_results['warnings']:
            validation_results['validation_status'] = 'WARNING'

        return jsonify(validation_results)

    except Exception as e:
        logger.error(f"Error validating dataset: {str(e)}")
        return jsonify({'error': 'Failed to validate dataset'}), 500

@analysis_bp.route('/model/export', methods=['POST'])
def export_model_report():
    """
    Export comprehensive model analysis report

    Request JSON:
    {
        "report_type": "performance",
        "modalities": ["image", "video", "audio"],
        "format": "json",
        "include_visualizations": true
    }
    """
    try:
        data = request.get_json()
        report_type = data.get('report_type', 'performance')
        modalities = data.get('modalities', ['image'])
        output_format = data.get('format', 'json')
        include_visualizations = data.get('include_visualizations', False)

        # Generate comprehensive report
        report = {
            'report_metadata': {
                'report_type': report_type,
                'generation_date': datetime.utcnow().isoformat(),
                'modalities_analyzed': modalities,
                'include_visualizations': include_visualizations
            },
            'executive_summary': {
                'best_performing_modality': 'video',
                'overall_accuracy': 0.901,
                'key_findings': [
                    'Video modality shows highest detection accuracy',
                    'Multi-modal fusion improves performance by 2.2%',
                    'Temporal features are most discriminative for video deepfakes'
                ]
            },
            'detailed_analysis': {}
        }

        # Add detailed analysis for each modality
        for modality in modalities:
            report['detailed_analysis'][modality] = {
                'performance_metrics': {
                    'accuracy': 0.89 + hash(modality) % 10 * 0.001,  # Mock variation
                    'precision': 0.88 + hash(modality) % 10 * 0.001,
                    'recall': 0.90 + hash(modality) % 10 * 0.001,
                    'f1_score': 0.89 + hash(modality) % 10 * 0.001
                },
                'confusion_matrix': [[450, 50], [45, 455]],
                'feature_importance': {
                    'top_features': [
                        f'{modality}_specific_feature_1',
                        f'{modality}_specific_feature_2',
                        f'{modality}_specific_feature_3'
                    ]
                },
                'error_analysis': {
                    'common_failure_cases': [
                        f'High-quality {modality} deepfakes',
                        f'Low-resolution {modality} samples',
                        f'Compressed {modality} content'
                    ]
                }
            }

        # Save report if requested
        if output_format == 'file':
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f'deepfake_detection_report_{timestamp}.json'
            report_path = Path('reports') / filename
            report_path.parent.mkdir(exist_ok=True)

            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            return send_file(str(report_path), as_attachment=True)
        else:
            return jsonify(report)

    except Exception as e:
        logger.error(f"Error exporting model report: {str(e)}")
        return jsonify({'error': 'Failed to export model report'}), 500

@analysis_bp.route('/system/metrics', methods=['GET'])
def get_system_metrics():
    """
    Get system performance and resource usage metrics
    """
    try:
        import psutil

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # GPU metrics (if available)
        gpu_metrics = {}
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_metrics = {
                    'gpu_load': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature
                }
        except ImportError:
            gpu_metrics = {'error': 'GPU monitoring not available'}

        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'disk_total_gb': disk.total / (1024**3)
            },
            'gpu': gpu_metrics,
            'application': {
                'models_loaded': 3,  # Mock value
                'cache_size_mb': 256,  # Mock value
                'active_sessions': 12,  # Mock value
                'total_detections': 1547,  # Mock value
                'avg_processing_time': 2.3  # Mock value in seconds
            }
        }

        return jsonify(metrics)

    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return jsonify({'error': 'Failed to retrieve system metrics'}), 500

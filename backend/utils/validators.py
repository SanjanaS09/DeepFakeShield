"""
Validation Utilities for Multi-Modal Deepfake Detection
Comprehensive validation for inputs, models, data, and system components
Includes schema validation, data integrity checks, and model validation
"""

import numpy as np
import torch
import logging
import re
import json
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class InputValidator:
    """
    Validator for user inputs and API requests
    Validates data types, ranges, formats, and security
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize Input Validator

        Args:
            strict_mode: Enable strict validation rules
        """
        self.strict_mode = strict_mode

        # Initialize validation schemas
        self._init_validation_schemas()

        logger.info("Initialized InputValidator")

    def _init_validation_schemas(self):
        """Initialize validation schemas for different input types"""

        # API request schemas
        self.api_schemas = {
            'detection_request': {
                'type': 'object',
                'required': ['file_type'],
                'properties': {
                    'file_type': {'type': 'string', 'enum': ['image', 'video', 'audio']},
                    'model_type': {'type': 'string', 'enum': ['xception', 'efficientnet', 'i3d', 'slowfast', 'ecapa_tdnn', 'wav2vec2', 'fusion']},
                    'confidence_threshold': {'type': 'number', 'minimum': 0.0, 'maximum': 1.0},
                    'batch_size': {'type': 'integer', 'minimum': 1, 'maximum': 32},
                    'enable_xai': {'type': 'boolean'},
                    'output_format': {'type': 'string', 'enum': ['json', 'detailed']}
                }
            },
            'batch_detection_request': {
                'type': 'object',
                'required': ['files', 'file_type'],
                'properties': {
                    'files': {
                        'type': 'array',
                        'minItems': 1,
                        'maxItems': 100,
                        'items': {'type': 'string'}
                    },
                    'file_type': {'type': 'string', 'enum': ['image', 'video', 'audio']},
                    'model_type': {'type': 'string'},
                    'parallel_processing': {'type': 'boolean'}
                }
            }
        }

        # File validation schemas
        self.file_schemas = {
            'image_file': {
                'max_size_mb': 50,
                'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                'min_dimensions': (32, 32),
                'max_dimensions': (8192, 8192)
            },
            'video_file': {
                'max_size_mb': 500,
                'allowed_extensions': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
                'max_duration_seconds': 300,
                'min_fps': 1,
                'max_fps': 120
            },
            'audio_file': {
                'max_size_mb': 100,
                'allowed_extensions': ['.wav', '.mp3', '.aac', '.flac', '.ogg'],
                'max_duration_seconds': 600,
                'sample_rates': [8000, 16000, 22050, 44100, 48000]
            }
        }

    def validate_api_request(self, request_data: Dict[str, Any], 
                            schema_name: str) -> Dict[str, Any]:
        """
        Validate API request against schema

        Args:
            request_data: Request data to validate
            schema_name: Name of schema to validate against

        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_data': {}
        }

        try:
            if schema_name not in self.api_schemas:
                result['valid'] = False
                result['errors'].append(f"Unknown schema: {schema_name}")
                return result

            schema = self.api_schemas[schema_name]

            # Validate required fields
            if 'required' in schema:
                for field in schema['required']:
                    if field not in request_data:
                        result['errors'].append(f"Missing required field: {field}")
                        result['valid'] = False

            # Validate properties
            if 'properties' in schema:
                for field, field_schema in schema['properties'].items():
                    if field in request_data:
                        field_validation = self._validate_field(
                            request_data[field], field_schema, field
                        )

                        if not field_validation['valid']:
                            result['errors'].extend(field_validation['errors'])
                            result['valid'] = False

                        result['warnings'].extend(field_validation['warnings'])
                        result['sanitized_data'][field] = field_validation['value']

            # Copy non-schema fields if not in strict mode
            if not self.strict_mode:
                for field, value in request_data.items():
                    if field not in result['sanitized_data']:
                        result['sanitized_data'][field] = value

        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
            logger.error(f"API request validation failed: {e}")

        return result

    def _validate_field(self, value: Any, schema: Dict[str, Any], 
                       field_name: str) -> Dict[str, Any]:
        """Validate individual field against schema"""

        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'value': value
        }

        try:
            # Type validation
            if 'type' in schema:
                expected_type = schema['type']

                if expected_type == 'string' and not isinstance(value, str):
                    result['errors'].append(f"{field_name} must be a string")
                    result['valid'] = False
                elif expected_type == 'number' and not isinstance(value, (int, float)):
                    result['errors'].append(f"{field_name} must be a number")
                    result['valid'] = False
                elif expected_type == 'integer' and not isinstance(value, int):
                    result['errors'].append(f"{field_name} must be an integer")
                    result['valid'] = False
                elif expected_type == 'boolean' and not isinstance(value, bool):
                    result['errors'].append(f"{field_name} must be a boolean")
                    result['valid'] = False
                elif expected_type == 'array' and not isinstance(value, list):
                    result['errors'].append(f"{field_name} must be an array")
                    result['valid'] = False
                elif expected_type == 'object' and not isinstance(value, dict):
                    result['errors'].append(f"{field_name} must be an object")
                    result['valid'] = False

            # Enum validation
            if 'enum' in schema and value not in schema['enum']:
                result['errors'].append(f"{field_name} must be one of: {schema['enum']}")
                result['valid'] = False

            # Range validation for numbers
            if isinstance(value, (int, float)):
                if 'minimum' in schema and value < schema['minimum']:
                    result['errors'].append(f"{field_name} must be >= {schema['minimum']}")
                    result['valid'] = False

                if 'maximum' in schema and value > schema['maximum']:
                    result['errors'].append(f"{field_name} must be <= {schema['maximum']}")
                    result['valid'] = False

            # Array validation
            if isinstance(value, list):
                if 'minItems' in schema and len(value) < schema['minItems']:
                    result['errors'].append(f"{field_name} must have at least {schema['minItems']} items")
                    result['valid'] = False

                if 'maxItems' in schema and len(value) > schema['maxItems']:
                    result['errors'].append(f"{field_name} must have at most {schema['maxItems']} items")
                    result['valid'] = False

            # String validation
            if isinstance(value, str):
                # Length validation
                if 'minLength' in schema and len(value) < schema['minLength']:
                    result['errors'].append(f"{field_name} must be at least {schema['minLength']} characters")
                    result['valid'] = False

                if 'maxLength' in schema and len(value) > schema['maxLength']:
                    result['errors'].append(f"{field_name} must be at most {schema['maxLength']} characters")
                    result['valid'] = False

                # Pattern validation
                if 'pattern' in schema:
                    if not re.match(schema['pattern'], value):
                        result['errors'].append(f"{field_name} does not match required pattern")
                        result['valid'] = False

                # Sanitization
                result['value'] = value.strip()

        except Exception as e:
            result['errors'].append(f"Field validation error for {field_name}: {str(e)}")
            result['valid'] = False

        return result

    def validate_file_upload(self, file_info: Dict[str, Any], 
                           file_type: str) -> Dict[str, Any]:
        """
        Validate uploaded file

        Args:
            file_info: File information dictionary
            file_type: Expected file type ('image', 'video', 'audio')

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            if file_type not in self.file_schemas:
                result['errors'].append(f"Unsupported file type: {file_type}")
                result['valid'] = False
                return result

            schema = self.file_schemas[f"{file_type}_file"]

            # File size validation
            if 'size_bytes' in file_info:
                size_mb = file_info['size_bytes'] / (1024 * 1024)
                if size_mb > schema['max_size_mb']:
                    result['errors'].append(f"File too large: {size_mb:.1f}MB > {schema['max_size_mb']}MB")
                    result['valid'] = False

            # Extension validation
            if 'file_path' in file_info or 'filename' in file_info:
                filename = file_info.get('filename', file_info.get('file_path', ''))
                extension = Path(filename).suffix.lower()

                if extension not in schema['allowed_extensions']:
                    result['errors'].append(f"Unsupported file extension: {extension}")
                    result['valid'] = False

            # Type-specific validation
            if file_type == 'image':
                result = self._validate_image_specific(file_info, schema, result)
            elif file_type == 'video':
                result = self._validate_video_specific(file_info, schema, result)
            elif file_type == 'audio':
                result = self._validate_audio_specific(file_info, schema, result)

        except Exception as e:
            result['errors'].append(f"File validation error: {str(e)}")
            result['valid'] = False
            logger.error(f"File upload validation failed: {e}")

        return result

    def _validate_image_specific(self, file_info: Dict[str, Any], 
                                schema: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate image-specific properties"""

        # Dimension validation
        if 'width' in file_info and 'height' in file_info:
            width, height = file_info['width'], file_info['height']
            min_w, min_h = schema['min_dimensions']
            max_w, max_h = schema['max_dimensions']

            if width < min_w or height < min_h:
                result['errors'].append(f"Image too small: {width}x{height} < {min_w}x{min_h}")
                result['valid'] = False

            if width > max_w or height > max_h:
                result['errors'].append(f"Image too large: {width}x{height} > {max_w}x{max_h}")
                result['valid'] = False

        return result

    def _validate_video_specific(self, file_info: Dict[str, Any], 
                                schema: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video-specific properties"""

        # Duration validation
        if 'duration' in file_info:
            duration = file_info['duration']
            if duration > schema['max_duration_seconds']:
                result['errors'].append(f"Video too long: {duration}s > {schema['max_duration_seconds']}s")
                result['valid'] = False

        # FPS validation
        if 'fps' in file_info:
            fps = file_info['fps']
            if fps < schema['min_fps'] or fps > schema['max_fps']:
                result['errors'].append(f"Invalid FPS: {fps} (must be {schema['min_fps']}-{schema['max_fps']})")
                result['valid'] = False

        return result

    def _validate_audio_specific(self, file_info: Dict[str, Any], 
                                schema: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate audio-specific properties"""

        # Duration validation
        if 'duration' in file_info:
            duration = file_info['duration']
            if duration > schema['max_duration_seconds']:
                result['errors'].append(f"Audio too long: {duration}s > {schema['max_duration_seconds']}s")
                result['valid'] = False

        # Sample rate validation
        if 'sample_rate' in file_info:
            sample_rate = file_info['sample_rate']
            if sample_rate not in schema['sample_rates']:
                result['warnings'].append(f"Unusual sample rate: {sample_rate}Hz")

        return result

    def validate_model_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model parameters

        Args:
            params: Model parameters

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_params': {}
        }

        try:
            # Common parameter validation
            common_params = {
                'confidence_threshold': (float, 0.0, 1.0),
                'batch_size': (int, 1, 128),
                'device': (str, ['cpu', 'cuda', 'auto']),
                'num_workers': (int, 0, 16),
                'pin_memory': (bool, None),
                'use_fp16': (bool, None)
            }

            for param_name, param_config in common_params.items():
                if param_name in params:
                    param_type = param_config[0]

                    # Type check
                    if not isinstance(params[param_name], param_type):
                        result['errors'].append(f"{param_name} must be of type {param_type.__name__}")
                        result['valid'] = False
                        continue

                    # Range/enum check
                    if param_type in [int, float] and len(param_config) > 2:
                        min_val, max_val = param_config[1], param_config[2]
                        if params[param_name] < min_val or params[param_name] > max_val:
                            result['errors'].append(f"{param_name} must be between {min_val} and {max_val}")
                            result['valid'] = False
                            continue
                    elif param_type == str and isinstance(param_config[1], list):
                        if params[param_name] not in param_config[1]:
                            result['errors'].append(f"{param_name} must be one of: {param_config[1]}")
                            result['valid'] = False
                            continue

                    result['sanitized_params'][param_name] = params[param_name]

            # Model-specific validation
            if 'model_type' in params:
                model_validation = self._validate_model_type_params(params)
                result['errors'].extend(model_validation['errors'])
                result['warnings'].extend(model_validation['warnings'])
                if not model_validation['valid']:
                    result['valid'] = False
                result['sanitized_params'].update(model_validation['sanitized_params'])

        except Exception as e:
            result['errors'].append(f"Parameter validation error: {str(e)}")
            result['valid'] = False
            logger.error(f"Model parameter validation failed: {e}")

        return result

    def _validate_model_type_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model-type specific parameters"""

        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'sanitized_params': {}
        }

        model_type = params.get('model_type', '')

        # Image model parameters
        if model_type in ['xception', 'efficientnet', 'resnet']:
            image_params = {
                'input_size': (int, 224),
                'num_classes': (int, 2),
                'pretrained': (bool, True),
                'dropout_rate': (float, 0.0, 0.9)
            }

            for param_name, config in image_params.items():
                if param_name in params:
                    if not isinstance(params[param_name], config[0]):
                        result['errors'].append(f"{param_name} must be of type {config[0].__name__}")
                        result['valid'] = False
                    else:
                        result['sanitized_params'][param_name] = params[param_name]

        # Video model parameters
        elif model_type in ['i3d', 'slowfast', 'x3d']:
            video_params = {
                'num_frames': (int, 8, 64),
                'temporal_stride': (int, 1, 8),
                'spatial_size': (int, 224),
                'slow_fast_alpha': (float, 4.0)  # For SlowFast only
            }

            for param_name, config in video_params.items():
                if param_name in params:
                    param_type = config[0]
                    if not isinstance(params[param_name], param_type):
                        result['errors'].append(f"{param_name} must be of type {param_type.__name__}")
                        result['valid'] = False
                    elif len(config) > 2:  # Has range
                        min_val, max_val = config[1], config[2]
                        if params[param_name] < min_val or params[param_name] > max_val:
                            result['errors'].append(f"{param_name} must be between {min_val} and {max_val}")
                            result['valid'] = False

                    if result['valid'] or param_name not in result['errors']:
                        result['sanitized_params'][param_name] = params[param_name]

        # Audio model parameters
        elif model_type in ['ecapa_tdnn', 'wav2vec2']:
            audio_params = {
                'sample_rate': (int, [8000, 16000, 22050, 44100, 48000]),
                'window_length': (float, 0.01, 0.1),
                'hop_length': (float, 0.001, 0.05),
                'n_mels': (int, 40, 128),
                'n_fft': (int, 512, 4096)
            }

            for param_name, config in audio_params.items():
                if param_name in params:
                    param_type = config[0]
                    if not isinstance(params[param_name], param_type):
                        result['errors'].append(f"{param_name} must be of type {param_type.__name__}")
                        result['valid'] = False
                    elif isinstance(config[1], list):  # Enum
                        if params[param_name] not in config[1]:
                            result['errors'].append(f"{param_name} must be one of: {config[1]}")
                            result['valid'] = False
                    elif len(config) > 2:  # Range
                        min_val, max_val = config[1], config[2]
                        if params[param_name] < min_val or params[param_name] > max_val:
                            result['errors'].append(f"{param_name} must be between {min_val} and {max_val}")
                            result['valid'] = False

                    if param_name not in [e.split()[0] for e in result['errors']]:
                        result['sanitized_params'][param_name] = params[param_name]

        return result


class ModelValidator:
    """
    Validator for machine learning models and their outputs
    Validates model architecture, weights, and prediction outputs
    """

    def __init__(self):
        """Initialize Model Validator"""
        logger.info("Initialized ModelValidator")

    def validate_model_architecture(self, model: Any, 
                                  expected_type: str = None) -> Dict[str, Any]:
        """
        Validate model architecture

        Args:
            model: Model to validate
            expected_type: Expected model type

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }

        try:
            # PyTorch model validation
            if hasattr(model, 'state_dict'):  # PyTorch model
                result['model_info']['framework'] = 'pytorch'
                result['model_info']['parameters'] = sum(p.numel() for p in model.parameters())
                result['model_info']['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

                # Check if model is in appropriate mode
                if hasattr(model, 'training'):
                    result['model_info']['training_mode'] = model.training
                    if model.training:
                        result['warnings'].append("Model is in training mode, consider setting to eval()")

                # Check device
                if hasattr(model, 'parameters'):
                    params = list(model.parameters())
                    if params:
                        device = str(params[0].device)
                        result['model_info']['device'] = device

            # TensorFlow/Keras model validation
            elif hasattr(model, 'summary'):  # Keras model
                result['model_info']['framework'] = 'tensorflow'
                try:
                    result['model_info']['parameters'] = model.count_params()
                except:
                    result['warnings'].append("Could not count model parameters")

            # Generic model validation
            else:
                result['model_info']['framework'] = 'unknown'
                result['warnings'].append("Unknown model framework")

            # Expected type validation
            if expected_type:
                model_type_valid = self._validate_model_type(model, expected_type)
                if not model_type_valid:
                    result['errors'].append(f"Model does not match expected type: {expected_type}")
                    result['valid'] = False

            # Architecture-specific checks
            arch_validation = self._validate_architecture_specific(model)
            result['errors'].extend(arch_validation['errors'])
            result['warnings'].extend(arch_validation['warnings'])
            if not arch_validation['valid']:
                result['valid'] = False

        except Exception as e:
            result['errors'].append(f"Model architecture validation error: {str(e)}")
            result['valid'] = False
            logger.error(f"Model architecture validation failed: {e}")

        return result

    def _validate_model_type(self, model: Any, expected_type: str) -> bool:
        """Validate model matches expected type"""

        try:
            model_class_name = model.__class__.__name__.lower()

            # Define type mappings
            type_mappings = {
                'xception': ['xception'],
                'efficientnet': ['efficientnet'],
                'resnet': ['resnet'],
                'i3d': ['i3d', 'inflated3d'],
                'slowfast': ['slowfast'],
                'x3d': ['x3d'],
                'ecapa_tdnn': ['ecapa', 'tdnn'],
                'wav2vec2': ['wav2vec', 'wav2vec2']
            }

            if expected_type.lower() in type_mappings:
                expected_patterns = type_mappings[expected_type.lower()]
                return any(pattern in model_class_name for pattern in expected_patterns)

            return True  # Unknown type, assume valid

        except Exception:
            return True  # Error in validation, assume valid

    def _validate_architecture_specific(self, model: Any) -> Dict[str, Any]:
        """Validate architecture-specific requirements"""

        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        try:
            # Check for required methods
            required_methods = ['forward', '__call__']
            for method in required_methods:
                if not hasattr(model, method):
                    result['warnings'].append(f"Model missing recommended method: {method}")

            # Check for common issues
            if hasattr(model, 'parameters'):
                params = list(model.parameters())
                if not params:
                    result['warnings'].append("Model has no parameters")

                # Check for NaN or inf values
                for i, param in enumerate(params):
                    if torch.isnan(param).any():
                        result['errors'].append(f"Parameter {i} contains NaN values")
                        result['valid'] = False
                    if torch.isinf(param).any():
                        result['errors'].append(f"Parameter {i} contains infinite values")
                        result['valid'] = False

        except Exception as e:
            result['warnings'].append(f"Architecture validation warning: {str(e)}")

        return result

    def validate_model_output(self, output: Any, expected_shape: Optional[Tuple] = None,
                            output_type: str = 'classification') -> Dict[str, Any]:
        """
        Validate model output

        Args:
            output: Model output to validate
            expected_shape: Expected output shape
            output_type: Type of output ('classification', 'regression', 'features')

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'output_info': {}
        }

        try:
            # Convert to numpy if tensor
            if isinstance(output, torch.Tensor):
                output_array = output.detach().cpu().numpy()
                result['output_info']['framework'] = 'pytorch'
            elif hasattr(output, 'numpy'):  # TensorFlow tensor
                output_array = output.numpy()
                result['output_info']['framework'] = 'tensorflow'
            else:
                output_array = np.array(output)
                result['output_info']['framework'] = 'numpy'

            # Basic shape validation
            result['output_info']['shape'] = output_array.shape
            result['output_info']['dtype'] = str(output_array.dtype)
            result['output_info']['size'] = output_array.size

            if expected_shape and output_array.shape != expected_shape:
                result['errors'].append(f"Output shape {output_array.shape} does not match expected {expected_shape}")
                result['valid'] = False

            # Check for invalid values
            if np.isnan(output_array).any():
                result['errors'].append("Output contains NaN values")
                result['valid'] = False

            if np.isinf(output_array).any():
                result['errors'].append("Output contains infinite values")
                result['valid'] = False

            # Output type specific validation
            if output_type == 'classification':
                result = self._validate_classification_output(output_array, result)
            elif output_type == 'regression':
                result = self._validate_regression_output(output_array, result)
            elif output_type == 'features':
                result = self._validate_feature_output(output_array, result)

        except Exception as e:
            result['errors'].append(f"Output validation error: {str(e)}")
            result['valid'] = False
            logger.error(f"Model output validation failed: {e}")

        return result

    def _validate_classification_output(self, output: np.ndarray, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate classification output"""

        # Check if probabilities sum to 1 (for softmax output)
        if output.ndim == 1 or (output.ndim == 2 and output.shape[0] == 1):
            probs = output.flatten()
            prob_sum = np.sum(probs)

            if abs(prob_sum - 1.0) > 0.01:
                result['warnings'].append(f"Probabilities don't sum to 1: {prob_sum:.4f}")

            if np.any(probs < 0) or np.any(probs > 1):
                result['warnings'].append("Probabilities outside [0,1] range")

            result['output_info']['probability_sum'] = float(prob_sum)
            result['output_info']['max_probability'] = float(np.max(probs))
            result['output_info']['min_probability'] = float(np.min(probs))

        return result

    def _validate_regression_output(self, output: np.ndarray, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regression output"""

        result['output_info']['min_value'] = float(np.min(output))
        result['output_info']['max_value'] = float(np.max(output))
        result['output_info']['mean_value'] = float(np.mean(output))
        result['output_info']['std_value'] = float(np.std(output))

        # Check for reasonable ranges
        if np.abs(np.min(output)) > 1e6 or np.abs(np.max(output)) > 1e6:
            result['warnings'].append("Output values are very large")

        return result

    def _validate_feature_output(self, output: np.ndarray, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature output"""

        # Check feature statistics
        if output.ndim >= 2:
            result['output_info']['feature_dim'] = output.shape[-1]
            result['output_info']['batch_size'] = output.shape[0] if output.ndim >= 2 else 1

            # Feature diversity
            feature_var = np.var(output, axis=0) if output.ndim >= 2 else np.var(output)
            result['output_info']['mean_feature_variance'] = float(np.mean(feature_var))

            # Check for dead features (always zero)
            if output.ndim >= 2:
                dead_features = np.sum(np.all(output == 0, axis=0))
                if dead_features > 0:
                    result['warnings'].append(f"{dead_features} features are always zero")

        return result


class DataValidator:
    """
    Validator for datasets and data integrity
    Validates data format, distribution, and quality
    """

    def __init__(self):
        """Initialize Data Validator"""
        logger.info("Initialized DataValidator")

    def validate_dataset_structure(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate dataset directory structure

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'structure_info': {}
        }

        try:
            dataset_path = Path(dataset_path)

            if not dataset_path.exists():
                result['errors'].append(f"Dataset path does not exist: {dataset_path}")
                result['valid'] = False
                return result

            # Expected structure: dataset/{modality}/{split}/{class}/
            modalities = ['image', 'video', 'audio']
            splits = ['train', 'validation', 'test']
            classes = ['FAKE', 'REAL']

            structure_info = {}

            for modality in modalities:
                modality_path = dataset_path / modality

                if modality_path.exists():
                    structure_info[modality] = {}

                    for split in splits:
                        split_path = modality_path / split

                        if split_path.exists():
                            structure_info[modality][split] = {}

                            for class_name in classes:
                                class_path = split_path / class_name

                                if class_path.exists():
                                    # Count files
                                    files = list(class_path.glob('*'))
                                    file_count = len([f for f in files if f.is_file()])
                                    structure_info[modality][split][class_name] = file_count
                                else:
                                    result['warnings'].append(f"Missing class directory: {class_path}")
                                    structure_info[modality][split][class_name] = 0
                        else:
                            result['warnings'].append(f"Missing split directory: {split_path}")
                            structure_info[modality][split] = {cls: 0 for cls in classes}
                else:
                    result['warnings'].append(f"Missing modality directory: {modality_path}")
                    structure_info[modality] = {split: {cls: 0 for cls in classes} for split in splits}

            result['structure_info'] = structure_info

            # Validate class balance
            balance_validation = self._validate_class_balance(structure_info)
            result['warnings'].extend(balance_validation['warnings'])

            # Validate split sizes
            split_validation = self._validate_split_sizes(structure_info)
            result['warnings'].extend(split_validation['warnings'])

        except Exception as e:
            result['errors'].append(f"Dataset structure validation error: {str(e)}")
            result['valid'] = False
            logger.error(f"Dataset structure validation failed: {e}")

        return result

    def _validate_class_balance(self, structure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate class balance in dataset"""

        result = {
            'warnings': []
        }

        try:
            for modality, modality_data in structure_info.items():
                for split, split_data in modality_data.items():
                    fake_count = split_data.get('FAKE', 0)
                    real_count = split_data.get('REAL', 0)

                    if fake_count + real_count > 0:
                        # Calculate imbalance ratio
                        min_count = min(fake_count, real_count)
                        max_count = max(fake_count, real_count)

                        if min_count == 0:
                            result['warnings'].append(f"{modality}/{split}: One class has no samples")
                        else:
                            imbalance_ratio = max_count / min_count
                            if imbalance_ratio > 3.0:
                                result['warnings'].append(f"{modality}/{split}: Severe class imbalance (ratio: {imbalance_ratio:.1f})")
                            elif imbalance_ratio > 1.5:
                                result['warnings'].append(f"{modality}/{split}: Moderate class imbalance (ratio: {imbalance_ratio:.1f})")

        except Exception as e:
            result['warnings'].append(f"Class balance validation error: {str(e)}")

        return result

    def _validate_split_sizes(self, structure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset split sizes"""

        result = {
            'warnings': []
        }

        try:
            for modality, modality_data in structure_info.items():
                # Calculate total samples per split
                split_totals = {}
                for split in ['train', 'validation', 'test']:
                    split_data = modality_data.get(split, {})
                    total = sum(split_data.values())
                    split_totals[split] = total

                total_samples = sum(split_totals.values())

                if total_samples > 0:
                    # Calculate split ratios
                    train_ratio = split_totals['train'] / total_samples
                    val_ratio = split_totals['validation'] / total_samples
                    test_ratio = split_totals['test'] / total_samples

                    # Check for reasonable split ratios
                    if train_ratio < 0.6:
                        result['warnings'].append(f"{modality}: Training set is small ({train_ratio:.1%})")

                    if val_ratio < 0.1:
                        result['warnings'].append(f"{modality}: Validation set is small ({val_ratio:.1%})")

                    if test_ratio < 0.1:
                        result['warnings'].append(f"{modality}: Test set is small ({test_ratio:.1%})")

                    # Warn about very small datasets
                    if total_samples < 100:
                        result['warnings'].append(f"{modality}: Very small dataset ({total_samples} total samples)")

        except Exception as e:
            result['warnings'].append(f"Split size validation error: {str(e)}")

        return result

    def validate_data_quality(self, data_samples: List[Any], 
                            data_type: str = 'image') -> Dict[str, Any]:
        """
        Validate data quality

        Args:
            data_samples: List of data samples to validate
            data_type: Type of data ('image', 'video', 'audio')

        Returns:
            Validation result
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_metrics': {}
        }

        try:
            if not data_samples:
                result['errors'].append("No data samples provided")
                result['valid'] = False
                return result

            if data_type == 'image':
                quality_validation = self._validate_image_quality(data_samples)
            elif data_type == 'video':
                quality_validation = self._validate_video_quality(data_samples)
            elif data_type == 'audio':
                quality_validation = self._validate_audio_quality(data_samples)
            else:
                result['errors'].append(f"Unsupported data type: {data_type}")
                result['valid'] = False
                return result

            result['quality_metrics'] = quality_validation['metrics']
            result['warnings'].extend(quality_validation['warnings'])

        except Exception as e:
            result['errors'].append(f"Data quality validation error: {str(e)}")
            result['valid'] = False
            logger.error(f"Data quality validation failed: {e}")

        return result

    def _validate_image_quality(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Validate image data quality"""

        result = {
            'metrics': {},
            'warnings': []
        }

        try:
            # Collect statistics
            shapes = [img.shape for img in images]
            sizes = [img.size for img in images]
            dtypes = [str(img.dtype) for img in images]

            # Shape consistency
            unique_shapes = set(shapes)
            result['metrics']['unique_shapes'] = len(unique_shapes)
            result['metrics']['most_common_shape'] = max(unique_shapes, key=shapes.count) if unique_shapes else None

            if len(unique_shapes) > 1:
                result['warnings'].append(f"Inconsistent image shapes: {len(unique_shapes)} different shapes found")

            # Data type consistency
            unique_dtypes = set(dtypes)
            result['metrics']['unique_dtypes'] = len(unique_dtypes)

            if len(unique_dtypes) > 1:
                result['warnings'].append(f"Inconsistent data types: {unique_dtypes}")

            # Quality metrics
            if images:
                # Sample a few images for quality analysis
                sample_size = min(10, len(images))
                sample_indices = np.linspace(0, len(images) - 1, sample_size, dtype=int)

                qualities = []
                for idx in sample_indices:
                    img = images[idx]

                    # Convert to grayscale for analysis
                    if len(img.shape) == 3:
                        gray = np.mean(img, axis=2)
                    else:
                        gray = img

                    # Simple quality metrics
                    sharpness = np.var(np.gradient(gray))  # Gradient variance
                    contrast = np.std(gray)  # Standard deviation
                    brightness = np.mean(gray)  # Mean brightness

                    qualities.append({
                        'sharpness': sharpness,
                        'contrast': contrast,
                        'brightness': brightness
                    })

                # Average quality metrics
                if qualities:
                    result['metrics']['average_sharpness'] = np.mean([q['sharpness'] for q in qualities])
                    result['metrics']['average_contrast'] = np.mean([q['contrast'] for q in qualities])
                    result['metrics']['average_brightness'] = np.mean([q['brightness'] for q in qualities])

        except Exception as e:
            result['warnings'].append(f"Image quality analysis error: {str(e)}")

        return result

    def _validate_video_quality(self, videos: List[List[np.ndarray]]) -> Dict[str, Any]:
        """Validate video data quality"""

        result = {
            'metrics': {},
            'warnings': []
        }

        try:
            # Video-specific metrics
            frame_counts = [len(video) for video in videos]
            result['metrics']['min_frames'] = min(frame_counts) if frame_counts else 0
            result['metrics']['max_frames'] = max(frame_counts) if frame_counts else 0
            result['metrics']['avg_frames'] = np.mean(frame_counts) if frame_counts else 0

            # Frame consistency within videos
            inconsistent_videos = 0
            for video in videos:
                if video:
                    frame_shapes = [frame.shape for frame in video]
                    if len(set(frame_shapes)) > 1:
                        inconsistent_videos += 1

            if inconsistent_videos > 0:
                result['warnings'].append(f"{inconsistent_videos} videos have inconsistent frame shapes")

            result['metrics']['videos_with_inconsistent_frames'] = inconsistent_videos

        except Exception as e:
            result['warnings'].append(f"Video quality analysis error: {str(e)}")

        return result

    def _validate_audio_quality(self, audio_samples: List[np.ndarray]) -> Dict[str, Any]:
        """Validate audio data quality"""

        result = {
            'metrics': {},
            'warnings': []
        }

        try:
            # Audio-specific metrics
            lengths = [len(audio) for audio in audio_samples]
            result['metrics']['min_length'] = min(lengths) if lengths else 0
            result['metrics']['max_length'] = max(lengths) if lengths else 0
            result['metrics']['avg_length'] = np.mean(lengths) if lengths else 0

            # Audio quality metrics
            if audio_samples:
                sample_size = min(5, len(audio_samples))
                sample_indices = np.linspace(0, len(audio_samples) - 1, sample_size, dtype=int)

                snr_estimates = []
                for idx in sample_indices:
                    audio = audio_samples[idx]

                    # Simple SNR estimation
                    signal_power = np.mean(audio ** 2)
                    # Estimate noise as high-frequency content
                    high_freq = np.diff(audio, n=2)  # Second derivative
                    noise_power = np.mean(high_freq ** 2) if len(high_freq) > 0 else 0

                    if noise_power > 0:
                        snr = 10 * np.log10(signal_power / noise_power)
                        snr_estimates.append(snr)

                if snr_estimates:
                    result['metrics']['average_snr'] = np.mean(snr_estimates)

        except Exception as e:
            result['warnings'].append(f"Audio quality analysis error: {str(e)}")

        return result

validator = InputValidator()

def validate_file_type(file_info, file_type):
    return validator.validate_file_upload(file_info, file_type)

def validate_file_size(file_info, file_type):
    return validator.validate_file_upload(file_info, file_type)

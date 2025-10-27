"""
Pytest configuration for Multi-Modal Deepfake Detection System tests
Contains fixtures, configuration, and test utilities
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import soundfile as sf
from io import BytesIO

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / 'test_data'
TEMP_DIR = None

@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """Setup test environment"""
    global TEMP_DIR

    # Create temporary directory for test files
    TEMP_DIR = Path(tempfile.mkdtemp(prefix='deepfake_test_'))

    # Create test data directory structure
    test_dirs = [
        'images/real',
        'images/fake', 
        'videos/real',
        'videos/fake',
        'audio/real',
        'audio/fake',
        'models',
        'configs',
        'outputs'
    ]

    for dir_path in test_dirs:
        (TEMP_DIR / dir_path).mkdir(parents=True, exist_ok=True)

    # Set environment variables
    os.environ['TEST_MODE'] = 'true'
    os.environ['TEST_DATA_DIR'] = str(TEMP_DIR)

    yield TEMP_DIR

    # Cleanup
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

@pytest.fixture
def temp_dir():
    """Get temporary directory for test files"""
    return TEMP_DIR

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Save to temp file
    img_path = TEMP_DIR / 'test_image.jpg'
    img.save(img_path)

    return img_path

@pytest.fixture
def sample_audio():
    """Create a sample test audio file"""
    duration = 3  # seconds
    sample_rate = 16000
    samples = np.random.randn(duration * sample_rate).astype(np.float32)

    # Save to temp file
    audio_path = TEMP_DIR / 'test_audio.wav'
    sf.write(audio_path, samples, sample_rate)

    return audio_path

@pytest.fixture
def sample_config():
    """Create a sample configuration file"""
    config_content = """
# Sample test configuration
model:
  name: "test_model"
  architecture: "xception"
  num_classes: 2

dataset:
  name: "test_dataset"
  data_root: "/tmp/test_data"

training:
  batch_size: 8
  num_epochs: 1
  learning_rate: 0.001
"""

    config_path = TEMP_DIR / 'test_config.yaml'
    with open(config_path, 'w') as f:
        f.write(config_content)

    return config_path

@pytest.fixture
def mock_model():
    """Create a mock PyTorch model for testing"""
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    return MockModel()

@pytest.fixture
def sample_batch():
    """Create a sample batch of data"""
    batch_size = 4
    input_dim = 10

    inputs = torch.randn(batch_size, input_dim)
    targets = torch.randint(0, 2, (batch_size,))

    return inputs, targets

@pytest.fixture
def api_client():
    """Create API client for testing"""
    from fastapi.testclient import TestClient

    # This would import your actual FastAPI app
    # from main import app
    # client = TestClient(app)

    # For now, return a mock client
    class MockClient:
        def get(self, url, **kwargs):
            class MockResponse:
                status_code = 200
                def json(self):
                    return {"status": "ok"}
            return MockResponse()

        def post(self, url, **kwargs):
            class MockResponse:
                status_code = 200
                def json(self):
                    return {"request_id": "test_123"}
            return MockResponse()

    return MockClient()

@pytest.fixture
def database_session():
    """Create database session for testing"""
    # This would create a test database session
    # For now, return a mock session
    class MockSession:
        def add(self, obj):
            pass

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

        def query(self, model):
            class MockQuery:
                def filter(self, *args):
                    return self

                def first(self):
                    return None

                def all(self):
                    return []

            return MockQuery()

    return MockSession()

# Test markers
pytest.mark.slow = pytest.mark.slow
pytest.mark.integration = pytest.mark.integration
pytest.mark.unit = pytest.mark.unit
pytest.mark.api = pytest.mark.api

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests") 
    config.addinivalue_line("markers", "api: marks tests as API tests")

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test file names
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        else:
            item.add_marker(pytest.mark.unit)

# Custom assertions
def assert_valid_detection_result(result):
    """Assert that a detection result is valid"""
    assert 'decision' in result
    assert result['decision'] in ['real', 'deepfake', 'uncertain']

    assert 'confidence_score' in result
    assert 0 <= result['confidence_score'] <= 1

    assert 'processing_time' in result
    assert result['processing_time'] > 0

def assert_valid_analysis_result(result):
    """Assert that an analysis result is valid"""
    assert 'metadata_analysis' in result or 'quality_analysis' in result

    if 'quality_analysis' in result:
        quality = result['quality_analysis']
        assert 'overall_score' in quality
        assert 0 <= quality['overall_score'] <= 1

def assert_model_output_shape(output, expected_shape):
    """Assert that model output has expected shape"""
    if isinstance(output, torch.Tensor):
        assert output.shape == expected_shape
    elif isinstance(output, dict):
        for key, tensor in output.items():
            assert isinstance(tensor, torch.Tensor)

# Test utilities
class TestDataGenerator:
    """Utility class for generating test data"""

    @staticmethod
    def create_image(size=(224, 224, 3), save_path=None):
        """Create a test image"""
        img_array = np.random.randint(0, 255, size, dtype=np.uint8)
        img = Image.fromarray(img_array)

        if save_path:
            img.save(save_path)

        return img

    @staticmethod
    def create_audio(duration=3, sample_rate=16000, save_path=None):
        """Create a test audio file"""
        samples = np.random.randn(duration * sample_rate).astype(np.float32)

        if save_path:
            sf.write(save_path, samples, sample_rate)

        return samples

    @staticmethod
    def create_batch_data(batch_size=4, input_shape=(3, 224, 224)):
        """Create a batch of test data"""
        inputs = torch.randn(batch_size, *input_shape)
        targets = torch.randint(0, 2, (batch_size,))

        return inputs, targets

class MockModelTrainer:
    """Mock trainer for testing training workflows"""

    def __init__(self, config):
        self.config = config
        self.current_epoch = 0
        self.best_score = 0.0

    def train(self):
        """Mock training method"""
        for epoch in range(self.config.get('num_epochs', 1)):
            self.current_epoch = epoch
            # Simulate training progress
            pass

    def validate(self):
        """Mock validation method"""
        return {
            'accuracy': 0.85,
            'auc_roc': 0.90,
            'loss': 0.3
        }

    def save_checkpoint(self, path):
        """Mock checkpoint saving"""
        pass

# Error classes for testing
class TestError(Exception):
    """Base exception for test errors"""
    pass

class ModelLoadError(TestError):
    """Error loading model during testing"""
    pass

class DataLoadError(TestError):
    """Error loading data during testing"""
    pass

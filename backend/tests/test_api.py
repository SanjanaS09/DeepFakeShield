"""
API Tests for Multi-Modal Deepfake Detection System
Tests for REST API endpoints, authentication, and response validation
"""

import pytest
import requests
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from io import BytesIO
from PIL import Image
import soundfile as sf

# Test configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
TEST_API_KEY = os.getenv('TEST_API_KEY', 'test_api_key')
TEST_DATA_DIR = Path(__file__).parent.parent / 'test_data'

class TestAPIBase:
    """Base class for API tests"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        self.headers = {
            'Authorization': f'Bearer {TEST_API_KEY}',
            'Content-Type': 'application/json'
        }
        self.api_url = API_BASE_URL

    def create_test_image(self, size=(224, 224)):
        """Create a test image"""
        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        return buffer

    def create_test_audio(self, duration=3, sample_rate=16000):
        """Create a test audio file"""
        samples = np.random.randn(duration * sample_rate).astype(np.float32)
        buffer = BytesIO()
        sf.write(buffer, samples, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer


class TestHealthEndpoints(TestAPIBase):
    """Test health check endpoints"""

    def test_health_check(self):
        """Test basic health check endpoint"""
        response = requests.get(f'{self.api_url}/health')
        assert response.status_code == 200

        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'

    def test_detailed_health_check(self):
        """Test detailed health check endpoint"""
        response = requests.get(f'{self.api_url}/health/detailed')
        assert response.status_code == 200

        data = response.json()
        assert 'api' in data
        assert 'database' in data
        assert 'models' in data

        # Check that all services are healthy
        for service, status in data.items():
            if isinstance(status, dict):
                assert status.get('status') in ['healthy', 'warning']


class TestAuthenticationEndpoints(TestAPIBase):
    """Test authentication endpoints"""

    def test_authentication_required(self):
        """Test that authentication is required for protected endpoints"""
        # Remove authorization header
        headers = {'Content-Type': 'application/json'}

        response = requests.post(
            f'{self.api_url}/api/v1/detect/image',
            headers=headers,
            json={'image_url': 'http://example.com/image.jpg'}
        )

        assert response.status_code == 401
        assert 'authentication' in response.json().get('detail', '').lower()

    def test_invalid_api_key(self):
        """Test invalid API key handling"""
        headers = {
            'Authorization': 'Bearer invalid_key',
            'Content-Type': 'application/json'
        }

        response = requests.post(
            f'{self.api_url}/api/v1/detect/image',
            headers=headers,
            json={'image_url': 'http://example.com/image.jpg'}
        )

        assert response.status_code == 401


class TestImageDetectionEndpoints(TestAPIBase):
    """Test image deepfake detection endpoints"""

    def test_image_upload_detection(self):
        """Test image detection via file upload"""
        test_image = self.create_test_image()

        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        headers = {'Authorization': f'Bearer {TEST_API_KEY}'}

        response = requests.post(
            f'{self.api_url}/api/v1/detect/image/upload',
            headers=headers,
            files=files
        )

        assert response.status_code == 200

        data = response.json()
        assert 'request_id' in data
        assert 'status' in data
        assert data['status'] in ['processing', 'completed']

        if data['status'] == 'completed':
            assert 'result' in data
            self._validate_detection_result(data['result'])

    def test_image_url_detection(self):
        """Test image detection via URL"""
        payload = {
            'image_url': 'https://example.com/test-image.jpg',
            'options': {
                'models': ['xception'],
                'confidence_threshold': 0.5,
                'enable_xai': True
            }
        }

        response = requests.post(
            f'{self.api_url}/api/v1/detect/image/url',
            headers=self.headers,
            json=payload
        )

        # This might return 400 if URL is not accessible, which is fine for testing
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            assert 'request_id' in data

    def test_batch_image_detection(self):
        """Test batch image detection"""
        test_images = [self.create_test_image() for _ in range(3)]

        files = [
            ('files', (f'test{i}.jpg', img, 'image/jpeg'))
            for i, img in enumerate(test_images)
        ]

        headers = {'Authorization': f'Bearer {TEST_API_KEY}'}

        response = requests.post(
            f'{self.api_url}/api/v1/detect/image/batch',
            headers=headers,
            files=files
        )

        assert response.status_code == 200

        data = response.json()
        assert 'request_id' in data
        assert 'batch_size' in data
        assert data['batch_size'] == 3

    def _validate_detection_result(self, result):
        """Validate detection result structure"""
        assert 'decision' in result
        assert result['decision'] in ['real', 'deepfake', 'uncertain']

        assert 'confidence_score' in result
        assert 0 <= result['confidence_score'] <= 1

        assert 'processing_time' in result
        assert result['processing_time'] > 0

        if 'model_results' in result:
            assert isinstance(result['model_results'], dict)

        if 'explanation' in result:
            assert 'summary' in result['explanation']


class TestVideoDetectionEndpoints(TestAPIBase):
    """Test video deepfake detection endpoints"""

    def test_video_upload_detection(self):
        """Test video detection via file upload"""
        # Create a simple test video file (placeholder)
        # In a real test, you would create an actual video file
        test_video = BytesIO(b'fake video content')

        files = {'file': ('test.mp4', test_video, 'video/mp4')}
        headers = {'Authorization': f'Bearer {TEST_API_KEY}'}

        response = requests.post(
            f'{self.api_url}/api/v1/detect/video/upload',
            headers=headers,
            files=files
        )

        # This might fail with actual validation, which is expected
        assert response.status_code in [200, 400, 422]


class TestAudioDetectionEndpoints(TestAPIBase):
    """Test audio deepfake detection endpoints"""

    def test_audio_upload_detection(self):
        """Test audio detection via file upload"""
        test_audio = self.create_test_audio()

        files = {'file': ('test.wav', test_audio, 'audio/wav')}
        headers = {'Authorization': f'Bearer {TEST_API_KEY}'}

        response = requests.post(
            f'{self.api_url}/api/v1/detect/audio/upload',
            headers=headers,
            files=files
        )

        assert response.status_code == 200

        data = response.json()
        assert 'request_id' in data


class TestAnalysisEndpoints(TestAPIBase):
    """Test media analysis endpoints"""

    def test_image_analysis(self):
        """Test image analysis endpoint"""
        test_image = self.create_test_image()

        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        headers = {'Authorization': f'Bearer {TEST_API_KEY}'}

        payload = {
            'analysis_types': ['metadata', 'quality', 'content'],
            'include_thumbnails': False
        }

        response = requests.post(
            f'{self.api_url}/api/v1/analyze/image',
            headers=headers,
            files=files,
            data={'options': json.dumps(payload)}
        )

        assert response.status_code == 200

        data = response.json()
        assert 'request_id' in data


class TestStatusEndpoints(TestAPIBase):
    """Test request status endpoints"""

    def test_request_status(self):
        """Test request status endpoint"""
        # First create a request
        test_image = self.create_test_image()
        files = {'file': ('test.jpg', test_image, 'image/jpeg')}
        headers = {'Authorization': f'Bearer {TEST_API_KEY}'}

        response = requests.post(
            f'{self.api_url}/api/v1/detect/image/upload',
            headers=headers,
            files=files
        )

        if response.status_code == 200:
            request_id = response.json()['request_id']

            # Check status
            status_response = requests.get(
                f'{self.api_url}/api/v1/status/{request_id}',
                headers=self.headers
            )

            assert status_response.status_code == 200

            status_data = status_response.json()
            assert 'status' in status_data
            assert status_data['status'] in ['pending', 'processing', 'completed', 'failed']


class TestErrorHandling(TestAPIBase):
    """Test error handling"""

    def test_invalid_file_format(self):
        """Test invalid file format handling"""
        # Upload a text file as an image
        test_file = BytesIO(b'This is not an image')
        files = {'file': ('test.txt', test_file, 'text/plain')}
        headers = {'Authorization': f'Bearer {TEST_API_KEY}'}

        response = requests.post(
            f'{self.api_url}/api/v1/detect/image/upload',
            headers=headers,
            files=files
        )

        assert response.status_code == 400
        assert 'error' in response.json()

    def test_file_too_large(self):
        """Test file size limit handling"""
        # This is a mock test - in reality, you'd create a very large file
        large_image = self.create_test_image(size=(5000, 5000))
        files = {'file': ('large_test.jpg', large_image, 'image/jpeg')}
        headers = {'Authorization': f'Bearer {TEST_API_KEY}'}

        response = requests.post(
            f'{self.api_url}/api/v1/detect/image/upload',
            headers=headers,
            files=files
        )

        # Should either succeed or fail with file size error
        assert response.status_code in [200, 413]

    def test_malformed_request(self):
        """Test malformed request handling"""
        response = requests.post(
            f'{self.api_url}/api/v1/detect/image/url',
            headers=self.headers,
            json={'invalid_field': 'invalid_value'}
        )

        assert response.status_code == 422  # Validation error


class TestRateLimiting(TestAPIBase):
    """Test rate limiting"""

    def test_rate_limit(self):
        """Test API rate limiting"""
        # Make multiple rapid requests
        responses = []

        for _ in range(10):
            response = requests.get(f'{self.api_url}/health', headers=self.headers)
            responses.append(response.status_code)

        # Should have at least some successful responses
        assert 200 in responses

        # Might have rate limit responses (429)
        # This depends on your rate limiting configuration


class TestModelManagement(TestAPIBase):
    """Test model management endpoints"""

    def test_list_models(self):
        """Test listing available models"""
        response = requests.get(
            f'{self.api_url}/api/v1/models',
            headers=self.headers
        )

        assert response.status_code == 200

        data = response.json()
        assert 'models' in data
        assert isinstance(data['models'], list)

        if data['models']:
            model = data['models'][0]
            assert 'name' in model
            assert 'modality' in model
            assert 'status' in model

    def test_model_info(self):
        """Test getting model information"""
        # First get list of models
        models_response = requests.get(
            f'{self.api_url}/api/v1/models',
            headers=self.headers
        )

        if models_response.status_code == 200:
            models = models_response.json()['models']

            if models:
                model_name = models[0]['name']

                info_response = requests.get(
                    f'{self.api_url}/api/v1/models/{model_name}',
                    headers=self.headers
                )

                assert info_response.status_code == 200

                info_data = info_response.json()
                assert 'name' in info_data
                assert 'performance_metrics' in info_data


# Pytest configuration
@pytest.fixture(scope='session')
def api_server():
    """Start API server for testing if not already running"""
    # This would start your API server if needed
    # For now, assume it's already running
    yield
    # Cleanup code here if needed


def test_api_is_running():
    """Test that the API server is running"""
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.skip(f"API server not running at {API_BASE_URL}")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

"""
Middleware for Multi-Modal Deepfake Detection API
Handles authentication, rate limiting, logging, CORS, and request validation
"""

import time
import logging
import functools
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from flask import Flask, request, jsonify, g, current_app
from werkzeug.exceptions import TooManyRequests
import hashlib
import hmac

logger = logging.getLogger(__name__)

class RequestLogger:
    """Middleware for request/response logging"""

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize request logger with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_appcontext(self.teardown)

    def before_request(self):
        """Log incoming requests"""
        g.start_time = time.time()
        g.request_id = self.generate_request_id()

        # Log request details
        logger.info(f"[{g.request_id}] {request.method} {request.path} - "
                   f"Remote: {request.remote_addr} - "
                   f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}")

        # Log request body for POST requests (limited)
        if request.method == 'POST' and request.content_type:
            if 'application/json' in request.content_type:
                try:
                    body = request.get_json()
                    if body:
                        # Sanitize sensitive data
                        sanitized_body = self.sanitize_request_body(body)
                        logger.debug(f"[{g.request_id}] Request body: {sanitized_body}")
                except Exception as e:
                    logger.debug(f"[{g.request_id}] Could not parse JSON body: {e}")

    def after_request(self, response):
        """Log outgoing responses"""
        if hasattr(g, 'start_time') and hasattr(g, 'request_id'):
            duration = round((time.time() - g.start_time) * 1000, 2)

            logger.info(f"[{g.request_id}] Response: {response.status_code} - "
                       f"Duration: {duration}ms - Size: {response.content_length or 0} bytes")

            # Log error responses in detail
            if response.status_code >= 400:
                logger.warning(f"[{g.request_id}] Error response: {response.status_code} - "
                              f"Data: {response.get_data(as_text=True)[:200]}...")

        return response

    def teardown(self, exception):
        """Clean up request context"""
        if exception is not None:
            if hasattr(g, 'request_id'):
                logger.error(f"[{g.request_id}] Request failed with exception: {exception}")

    @staticmethod
    def generate_request_id() -> str:
        """Generate unique request ID"""
        timestamp = str(int(time.time() * 1000000))
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]

    @staticmethod
    def sanitize_request_body(body: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from request body for logging"""
        sensitive_keys = ['password', 'token', 'api_key', 'secret', 'auth']

        if isinstance(body, dict):
            sanitized = {}
            for key, value in body.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    sanitized[key] = '[REDACTED]'
                elif isinstance(value, dict):
                    sanitized[key] = RequestLogger.sanitize_request_body(value)
                else:
                    sanitized[key] = value
            return sanitized
        return body

class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self.requests = {}  # {client_ip: [(timestamp, endpoint), ...]}
        self.limits = {
            'default': {'requests': 100, 'window': 3600},  # 100 requests per hour
            '/api/detection': {'requests': 20, 'window': 3600},  # 20 detections per hour
            '/api/analysis': {'requests': 50, 'window': 3600}  # 50 analysis requests per hour
        }

        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize rate limiter with Flask app"""
        app.before_request(self.check_rate_limit)

    def check_rate_limit(self):
        """Check if request exceeds rate limit"""
        if not current_app.config.get('RATELIMIT_ENABLED', True):
            return

        client_ip = self.get_client_ip()
        endpoint = request.endpoint or request.path
        current_time = time.time()

        # Get appropriate limit
        limit_config = self.get_limit_config(endpoint)
        max_requests = limit_config['requests']
        window_seconds = limit_config['window']

        # Clean old requests
        self.cleanup_old_requests(client_ip, current_time, window_seconds)

        # Check current request count
        if client_ip in self.requests:
            request_count = len(self.requests[client_ip])
            if request_count >= max_requests:
                logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}: "
                              f"{request_count}/{max_requests} requests in {window_seconds}s")

                raise TooManyRequests(description={
                    'error': 'Rate limit exceeded',
                    'limit': max_requests,
                    'window_seconds': window_seconds,
                    'retry_after': window_seconds
                })

        # Record this request
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        self.requests[client_ip].append((current_time, endpoint))

    def get_client_ip(self) -> str:
        """Get client IP address"""
        # Check for forwarded headers (reverse proxy)
        if request.headers.get('X-Forwarded-For'):
            return request.headers['X-Forwarded-For'].split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers['X-Real-IP']
        else:
            return request.remote_addr or 'unknown'

    def get_limit_config(self, endpoint: str) -> Dict[str, int]:
        """Get rate limit configuration for endpoint"""
        for pattern, config in self.limits.items():
            if pattern in endpoint:
                return config
        return self.limits['default']

    def cleanup_old_requests(self, client_ip: str, current_time: float, window_seconds: int):
        """Remove requests outside the time window"""
        if client_ip in self.requests:
            cutoff_time = current_time - window_seconds
            self.requests[client_ip] = [
                (timestamp, endpoint) for timestamp, endpoint in self.requests[client_ip]
                if timestamp > cutoff_time
            ]

            # Remove empty entries
            if not self.requests[client_ip]:
                del self.requests[client_ip]

class SecurityHeaders:
    """Middleware for adding security headers"""

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize security headers with Flask app"""
        app.after_request(self.add_security_headers)

    def add_security_headers(self, response):
        """Add security headers to response"""
        # Content Security Policy
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "media-src 'self' blob:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none';"
        )

        response.headers['Content-Security-Policy'] = csp_policy
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Remove server information
        response.headers['Server'] = 'DeepfakeDetectionAPI/1.0'

        return response

class APIKeyAuth:
    """Simple API key authentication middleware"""

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self.protected_endpoints = [
            '/api/detection',
            '/api/analysis',
        ]

        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize API key auth with Flask app"""
        app.before_request(self.check_api_key)

    def check_api_key(self):
        """Check API key for protected endpoints"""
        # Skip auth for non-protected endpoints
        if not any(endpoint in request.path for endpoint in self.protected_endpoints):
            return

        # Skip auth if disabled in config
        if not current_app.config.get('API_KEY_REQUIRED', False):
            return

        # Check for API key in headers
        api_key = request.headers.get('X-API-Key') or request.headers.get('Authorization')

        if not api_key:
            return jsonify({
                'error': 'API key required',
                'message': 'Please provide API key in X-API-Key header'
            }), 401

        # Remove 'Bearer ' prefix if present
        if api_key.startswith('Bearer '):
            api_key = api_key[7:]

        # Validate API key
        if not self.validate_api_key(api_key):
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is invalid or expired'
            }), 403

    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key against configured keys"""
        valid_keys = current_app.config.get('VALID_API_KEYS', [])

        # If no keys configured, allow all (development mode)
        if not valid_keys:
            return True

        # Check against valid keys
        return api_key in valid_keys

class RequestValidator:
    """Middleware for request validation"""

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        self.max_file_size = 500 * 1024 * 1024  # 500MB

        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize request validator with Flask app"""
        app.before_request(self.validate_request)

    def validate_request(self):
        """Validate incoming requests"""
        # Check content length
        if request.content_length and request.content_length > self.max_file_size:
            return jsonify({
                'error': 'Request too large',
                'message': f'Maximum request size is {self.max_file_size // (1024*1024)}MB'
            }), 413

        # Validate file uploads
        if request.files:
            for file_key, file_obj in request.files.items():
                if not self.validate_uploaded_file(file_obj):
                    return jsonify({
                        'error': 'Invalid file',
                        'message': f'File {file_obj.filename} failed validation'
                    }), 400

    def validate_uploaded_file(self, file_obj) -> bool:
        """Validate uploaded file"""
        if not file_obj.filename:
            return False

        # Check file extension
        allowed_extensions = set()
        allowed_extensions.update(current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', set()))
        allowed_extensions.update(current_app.config.get('ALLOWED_VIDEO_EXTENSIONS', set()))
        allowed_extensions.update(current_app.config.get('ALLOWED_AUDIO_EXTENSIONS', set()))

        file_extension = file_obj.filename.rsplit('.', 1)[-1].lower()
        if file_extension not in allowed_extensions:
            logger.warning(f"Rejected file with extension: {file_extension}")
            return False

        return True

class CacheHeaders:
    """Middleware for cache control headers"""

    def __init__(self, app: Optional[Flask] = None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask):
        """Initialize cache headers with Flask app"""
        app.after_request(self.add_cache_headers)

    def add_cache_headers(self, response):
        """Add appropriate cache headers"""
        # No caching for API endpoints
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

        # Cache static files
        elif any(request.path.endswith(ext) for ext in ['.js', '.css', '.png', '.jpg', '.gif', '.ico']):
            response.headers['Cache-Control'] = 'public, max-age=86400'  # 24 hours

        return response

def setup_middleware(app: Flask):
    """Setup all middleware components"""
    logger.info("Setting up API middleware...")

    # Initialize middleware components
    RequestLogger(app)

    # Only enable rate limiting in production
    if not app.config.get('TESTING', False):
        RateLimiter(app)

    SecurityHeaders(app)
    CacheHeaders(app)

    # Only enable API key auth if configured
    if app.config.get('API_KEY_REQUIRED', False):
        APIKeyAuth(app)

    RequestValidator(app)

    # Error handlers
    @app.errorhandler(TooManyRequests)
    def handle_rate_limit_exceeded(e):
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.',
            'retry_after': e.description.get('retry_after', 3600) if isinstance(e.description, dict) else 3600
        }), 429

    @app.errorhandler(413)
    def handle_payload_too_large(e):
        return jsonify({
            'error': 'Payload too large',
            'message': 'The uploaded file or request is too large.'
        }), 413

    @app.errorhandler(415)
    def handle_unsupported_media_type(e):
        return jsonify({
            'error': 'Unsupported media type',
            'message': 'The uploaded file type is not supported.'
        }), 415

    @app.errorhandler(500)
    def handle_internal_error(e):
        logger.error(f"Internal server error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred. Please try again later.'
        }), 500

    logger.info("Middleware setup complete")

def require_auth(f: Callable) -> Callable:
    """Decorator for endpoints requiring authentication"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if API key is provided
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'Authentication required'}), 401

        # Validate API key (simplified)
        valid_keys = current_app.config.get('VALID_API_KEYS', [])
        if valid_keys and api_key not in valid_keys:
            return jsonify({'error': 'Invalid API key'}), 403

        return f(*args, **kwargs)

    return decorated_function

def rate_limit(max_requests: int, window_seconds: int):
    """Decorator for custom rate limiting"""
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # This would implement custom rate limiting
            # For now, just call the original function
            return f(*args, **kwargs)
        return decorated_function
    return decorator

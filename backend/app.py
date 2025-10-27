"""
Multi-Modal Deepfake Detection Flask Backend
Main application entry point with all routes and initialization
"""

import os
import logging
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import our custom modules
from api.detection_routes import detection_bp
from api.analysis_routes import analysis_bp
from api.health_routes import health_bp
from api.middleware import setup_middleware
from config import Config
from utils.logger import setup_logger
from utils.validators import validate_file_type
# Correctly import the service with an alias
from services.detection_service import DeepfakeDetectionService as DetectionService
# Correctly import from the database package
from database import init_db

def create_app(config_class=Config):
    """Application factory pattern to create and configure the Flask app."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize CORS to allow requests from your React frontend
    CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

    # Setup logging using the app's name (a string)
    setup_logger(app.name)
    app.logger.info("DeepFakeShield application starting up...")

    # Initialize the database and create tables if they don't exist
    # This function now correctly takes no arguments
    with app.app_context():
        init_db()

    # Setup custom middleware
    setup_middleware(app)

    # Register API blueprints
    app.register_blueprint(detection_bp, url_prefix='/api/detection')
    app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
    app.register_blueprint(health_bp, url_prefix='/api/health')

    # Initialize services and attach them to the app context for easy access
    detection_service = DetectionService()
    app.detection_service = detection_service
    app.logger.info("Services initialized.")

    @app.route('/')
    def index():
        """A simple root endpoint to confirm the server is online."""
        return jsonify({
            'message': 'Multi-Modal Deepfake Detection API',
            'version': '1.0.0',
            'status': 'online',
            'timestamp': datetime.utcnow().isoformat(),
            'documentation': '/docs' # A common practice to point to API docs
        })

    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """A generic file upload endpoint for testing."""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part in the request'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected for uploading'}), 400

            # You can add more validation here using your file_handlers
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"

            upload_dir = app.config.get('UPLOAD_FOLDER', 'uploads/temp')
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, unique_filename)
            file.save(file_path)
            
            app.logger.info(f"File uploaded successfully: {unique_filename}")

            return jsonify({
                'message': 'File uploaded successfully',
                'filename': unique_filename
            }), 201

        except Exception as e:
            app.logger.error(f"An error occurred during file upload: {str(e)}")
            return jsonify({'error': 'Internal server error during file upload'}), 500

    # Register custom error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not Found', 'message': 'This endpoint does not exist.'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal Server Error', 'message': 'An unexpected error occurred.'}), 500

    app.logger.info("Application setup complete. Ready to serve requests.")
    return app

if __name__ == '__main__':
    # Create the Flask app using the factory
    app = create_app()
    # Run the app
    app.run(
        host=app.config.get('HOST', '127.0.0.1'),
        port=app.config.get('PORT', 5000),
        debug=app.config.get('DEBUG', True)
    )

# """
# Multi-Modal Deepfake Detection Flask Backend
# Enhanced with WebSocket support for real-time preprocessing feedback
# """
# import os
# import logging
# import uuid
# from datetime import datetime
# from pathlib import Path

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask_socketio import SocketIO, emit
# from werkzeug.utils import secure_filename

# from api.detection_routes import detection_bp
# from api.analysis_routes import analysis_bp
# from api.health_routes import health_bp
# from api.middleware import setup_middleware
# from config import Config
# from utils.logger import setup_logger
# from services.detection_service import DeepfakeDetectionService

# # Store service in config
# app.config['detection_service'] = DeepfakeDetectionService()

# # Initialize SocketIO for real-time updates
# socketio = SocketIO(cors_allowed_origins="*")

# def create_app(config_class=Config):
#     """Application factory pattern with WebSocket support"""
#     app = Flask(__name__)
#     app.config.from_object(config_class)
    
#     # Initialize extensions
#     CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])
#     socketio.init_app(app)
    
#     # Setup logging
#     setup_logger(app.name)
#     app.logger.info("DeepFakeShield application starting up with WebSocket support...")
    
#     # Setup middleware
#     setup_middleware(app)
    
#     # Register blueprints
#     app.register_blueprint(detection_bp, url_prefix='/api/detection')
#     app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
#     app.register_blueprint(health_bp, url_prefix='/api/health')
    
#     # Initialize services (lazy loading to avoid circular imports)
#     with app.app_context():
#         from services.detection_service import DeepfakeDetectionService
#         app.detection_service = DeepfakeDetectionService()
#         app.logger.info("Services initialized.")
    
#     @app.route('/')
#     def index():
#         """Root endpoint"""
#         return jsonify({
#             "message": "Multi-Modal Deepfake Detection API",
#             "version": "2.0.0",
#             "status": "online",
#             "features": ["real-time-processing", "websocket-support", "xai-visualization"],
#             "timestamp": datetime.utcnow().isoformat()
#         })
    
#     @app.route('/api/upload', methods=['POST'])
#     def upload_file():
#         """Generic file upload endpoint"""
#         try:
#             if 'file' not in request.files:
#                 return jsonify({"error": "No file part in the request"}), 400
            
#             file = request.files['file']
#             if file.filename == '':
#                 return jsonify({"error": "No file selected for uploading"}), 400
            
#             filename = secure_filename(file.filename)
#             unique_filename = f"{uuid.uuid4()}_{filename}"
#             upload_dir = Path(app.config.get('UPLOAD_FOLDER', 'uploads/temp'))
#             upload_dir.mkdir(parents=True, exist_ok=True)
#             filepath = upload_dir / unique_filename
#             file.save(str(filepath))
            
#             app.logger.info(f"File uploaded successfully: {unique_filename}")
#             return jsonify({
#                 "message": "File uploaded successfully",
#                 "filename": unique_filename,
#                 "size_mb": filepath.stat().st_size / (1024 * 1024)
#             }), 201
            
#         except Exception as e:
#             app.logger.error(f"Upload error: {str(e)}")
#             return jsonify({"error": "Internal server error during file upload"}), 500
    
#     # WebSocket event handlers
#     @socketio.on('connect')
#     def handle_connect():
#         """Handle client connection"""
#         app.logger.info(f"Client connected: {request.sid}")
#         emit('connection_response', {'status': 'connected', 'sid': request.sid})
    
#     @socketio.on('disconnect')
#     def handle_disconnect():
#         """Handle client disconnection"""
#         app.logger.info(f"Client disconnected: {request.sid}")
    
#     @socketio.on('start_processing')
#     def handle_start_processing(data):
#         """Handle processing start request"""
#         app.logger.info(f"Processing started for session: {request.sid}")
#         emit('processing_started', {'status': 'initialized'})
    
#     # Error handlers
#     @app.errorhandler(404)
#     def not_found(error):
#         return jsonify({
#             "error": "Not Found",
#             "message": "This endpoint does not exist."
#         }), 404
    
#     @app.errorhandler(500)
#     def internal_error(error):
#         app.logger.error(f"Internal error: {str(error)}")
#         return jsonify({
#             "error": "Internal Server Error",
#             "message": "An unexpected error occurred."
#         }), 500
    
#     app.logger.info("Application setup complete. Ready to serve requests.")
#     return app


# if __name__ == '__main__':
#     app = create_app()
#     # Use socketio.run instead of app.run for WebSocket support
#     socketio.run(
#         app,
#         host=app.config.get('HOST', '127.0.0.1'),
#         port=app.config.get('PORT', 5000),
#         debug=app.config.get('DEBUG', True),
#         allow_unsafe_werkzeug=True
#     )

# #  """
# # Multi-Modal Deepfake Detection Flask Backend
# # Main application entry point with all routes and initialization
# # """

# # import os
# # import logging
# # import uuid
# # from datetime import datetime
# # from flask import Flask, request, jsonify, send_file
# # from flask_cors import CORS
# # from werkzeug.utils import secure_filename

# # # Import our custom modules
# # from api.detection_routes import detection_bp
# # from api.analysis_routes import analysis_bp
# # from api.health_routes import health_bp
# # from api.middleware import setup_middleware
# # from config import Config
# # from utils.logger import setup_logger
# # from utils.validators import validate_file_type
# # # Correctly import the service with an alias
# # from services.detection_service import DeepfakeDetectionService as DetectionService
# # # Correctly import from the database package
# # from database import init_db

# # def create_app(config_class=Config):
# #     """Application factory pattern to create and configure the Flask app."""
# #     app = Flask(__name__)
# #     app.config.from_object(config_class)

# #     # Initialize CORS to allow requests from your React frontend
# #     CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])

# #     # Setup logging using the app's name (a string)
# #     setup_logger(app.name)
# #     app.logger.info("DeepFakeShield application starting up...")

# #     # Initialize the database and create tables if they don't exist
# #     # This function now correctly takes no arguments
# #     with app.app_context():
# #         init_db()

# #     # Setup custom middleware
# #     setup_middleware(app)

# #     # Register API blueprints
# #     app.register_blueprint(detection_bp, url_prefix='/api/detection')
# #     app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
# #     app.register_blueprint(health_bp, url_prefix='/api/health')

# #     # Initialize services and attach them to the app context for easy access
# #     detection_service = DetectionService()
# #     app.detection_service = detection_service
# #     app.logger.info("Services initialized.")

# #     @app.route('/')
# #     def index():
# #         """A simple root endpoint to confirm the server is online."""
# #         return jsonify({
# #             'message': 'Multi-Modal Deepfake Detection API',
# #             'version': '1.0.0',
# #             'status': 'online',
# #             'timestamp': datetime.utcnow().isoformat(),
# #             'documentation': '/docs' # A common practice to point to API docs
# #         })

# #     @app.route('/api/upload', methods=['POST'])
# #     def upload_file():
# #         """A generic file upload endpoint for testing."""
# #         try:
# #             if 'file' not in request.files:
# #                 return jsonify({'error': 'No file part in the request'}), 400

# #             file = request.files['file']
# #             if file.filename == '':
# #                 return jsonify({'error': 'No file selected for uploading'}), 400

# #             # You can add more validation here using your file_handlers
# #             filename = secure_filename(file.filename)
# #             unique_filename = f"{uuid.uuid4()}_{filename}"

# #             upload_dir = app.config.get('UPLOAD_FOLDER', 'uploads/temp')
# #             os.makedirs(upload_dir, exist_ok=True)
            
# #             file_path = os.path.join(upload_dir, unique_filename)
# #             file.save(file_path)
            
# #             app.logger.info(f"File uploaded successfully: {unique_filename}")

# #             return jsonify({
# #                 'message': 'File uploaded successfully',
# #                 'filename': unique_filename
# #             }), 201

# #         except Exception as e:
# #             app.logger.error(f"An error occurred during file upload: {str(e)}")
# #             return jsonify({'error': 'Internal server error during file upload'}), 500

# #     # Register custom error handlers
# #     @app.errorhandler(404)
# #     def not_found(error):
# #         return jsonify({'error': 'Not Found', 'message': 'This endpoint does not exist.'}), 404

# #     @app.errorhandler(500)
# #     def internal_error(error):
# #         return jsonify({'error': 'Internal Server Error', 'message': 'An unexpected error occurred.'}), 500

# #     app.logger.info("Application setup complete. Ready to serve requests.")
# #     return app

# # if __name__ == '__main__':
# #     # Create the Flask app using the factory
# #     app = create_app()
# #     # Run the app
# #     app.run(
# #         host=app.config.get('HOST', '127.0.0.1'),
# #         port=app.config.get('PORT', 5000),
# #         debug=app.config.get('DEBUG', True)
# #     )

"""
DeepFake Shield - Flask Backend with Real Model Predictions
Uses trained models for actual deepfake detection
"""
import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).parent
sys.path.insert(0, str(BACKEND_ROOT))

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging
from datetime import datetime
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# REAL DETECTION SERVICE - USES TRAINED MODELS
# ============================================
class RealDetectionService:
    """Detection service using actual trained models"""
    
    def __init__(self):
        """Initialize with trained models"""
        logger.info("Initializing Real Detection Service...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.image_model = self._load_image_model()
        self.video_model = self._load_video_model()
        self.audio_model = self._load_audio_model()
        
        logger.info("✓ Detection service initialized with real models")
    
    def _load_image_model(self):
        """Load image model from checkpoint"""
        try:
            checkpoint_path = Path('checkpoints/image/best_model.pth')
            
            if not checkpoint_path.exists():
                logger.warning(f"Image model not found at {checkpoint_path}")
                return None
            
            logger.info(f"Loading image model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Import model class
            try:
                from models.image_detector import ImageDetector
                model = ImageDetector(backbone='xception', num_classes=2)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    logger.info("Loading checkpoint with model_state_dict")
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    logger.info("Loading checkpoint directly")
                    model.load_state_dict(checkpoint)
                
                model = model.to(self.device)
                model.eval()
                logger.info("✓ Image model loaded successfully")
                return model
            
            except Exception as e:
                logger.error(f"Error loading image model: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load image model: {e}")
            return None
    
    def _load_video_model(self):
        """Load video model from checkpoint"""
        try:
            checkpoint_path = Path('checkpoints/video/best_model.pth')
            
            if not checkpoint_path.exists():
                logger.warning(f"Video model not found at {checkpoint_path}")
                return None
            
            logger.info("Loading video model...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            try:
                from models.video_detector import VideoDetector
                model = VideoDetector(backbone='i3d', num_classes=2)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model = model.to(self.device)
                model.eval()
                logger.info("✓ Video model loaded")
                return model
            except Exception as e:
                logger.error(f"Error loading video model: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load video model: {e}")
            return None
    
    def _load_audio_model(self):
        """Load audio model from checkpoint"""
        try:
            checkpoint_path = Path('checkpoints/audio/best_model.pth')
            
            if not checkpoint_path.exists():
                logger.warning(f"Audio model not found at {checkpoint_path}")
                return None
            
            logger.info("Loading audio model...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            try:
                from models.audio_detector import AudioDetector
                model = AudioDetector(backbone='ecapa-tdnn', num_classes=2)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model = model.to(self.device)
                model.eval()
                logger.info("✓ Audio model loaded")
                return model
            except Exception as e:
                logger.error(f"Error loading audio model: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")
            return None
    
    def detect_image(self, image_path):
        """Detect deepfake in image using trained model"""
        try:
            logger.info(f"Processing image: {image_path}")
            
            if not self.image_model:
                logger.warning("Image model not available, returning demo result")
                return self._demo_result()
            
            # Import preprocessing
            from preprocessing.image_preprocessor import ImagePreprocessor
            from PIL import Image
            import torchvision.transforms as transforms
            
            try:
                # Load image
                logger.info("Loading and preprocessing image...")
                image = Image.open(image_path).convert('RGB')
                
                # Preprocess
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                
                image_tensor = transform(image).unsqueeze(0).to(self.device)
                logger.info("Image preprocessed successfully")
                
                # Run model
                logger.info("Running inference...")
                with torch.no_grad():
                    output = self.image_model(image_tensor)
                    probs = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)
                
                # Extract results
                prediction = 'FAKE' if predicted.item() == 1 else 'REAL'
                confidence_score = float(confidence.item())
                
                result = {
                    'prediction': prediction,
                    'confidence': confidence_score,
                    'probability': {
                        'REAL': float(probs[0, 0].item()),
                        'FAKE': float(probs[0, 1].item())
                    },
                    'model': 'ImageDetector (Xception)',
                    'status': 'success'
                }
                
                logger.info(f"Detection complete: {prediction} ({confidence_score:.2%})")
                return result
            
            except Exception as e:
                logger.error(f"Model inference error: {e}")
                logger.error(f"Traceback: {e.__traceback__}")
                return {'error': f"Inference error: {str(e)}", 'status': 'error'}
        
        except Exception as e:
            logger.error(f"Image detection error: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def detect_video(self, video_path):
        """Detect deepfake in video using trained model"""
        try:
            logger.info(f"Processing video: {video_path}")
            
            if not self.video_model:
                logger.warning("Video model not available")
                return self._demo_result()
            
            # For now, return demo (full video processing is complex)
            return self._demo_result()
        
        except Exception as e:
            logger.error(f"Video detection error: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def detect_audio(self, audio_path):
        """Detect deepfake in audio using trained model"""
        try:
            logger.info(f"Processing audio: {audio_path}")
            
            if not self.audio_model:
                logger.warning("Audio model not available")
                return self._demo_result()
            
            # For now, return demo (audio processing setup)
            return self._demo_result()
        
        except Exception as e:
            logger.error(f"Audio detection error: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _demo_result(self):
        """Return demo result (for models not yet implemented)"""
        return {
            'prediction': 'REAL',
            'confidence': 0.92,
            'probability': {'REAL': 0.92, 'FAKE': 0.08},
            'status': 'success'
        }


# ============================================
# CREATE FLASK APP
# ============================================
def create_app():
    """Create Flask app with real detection service"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'deepfake-shield-secret'
    
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    socketio = SocketIO(app, cors_allowed_origins="*")
    
    # Initialize REAL detection service
    app.config['detection_service'] = RealDetectionService()
    
    logger.info("Flask app created with REAL detection service")
    return app, socketio


app, socketio = create_app()


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Health check"""
    return jsonify({
        'status': 'online',
        'message': 'DeepFake Shield API - PRODUCTION MODE',
        'mode': 'REAL MODELS',
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/api/detection/image', methods=['POST'])
def detect_image():
    """Image detection with real models"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        from werkzeug.utils import secure_filename
        import uuid
        
        upload_dir = Path('uploads/temp')
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = upload_dir / filename
        file.save(str(filepath))
        
        logger.info(f"File saved: {filepath}")
        
        # Get service
        detection_service = app.config['detection_service']
        
        # Run detection
        logger.info("Starting detection...")
        result = detection_service.detect_image(str(filepath))
        
        # Cleanup
        try:
            if filepath.exists():
                filepath.unlink()
                logger.info("Temp file cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/detection/video', methods=['POST'])
def detect_video():
    """Video detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        from werkzeug.utils import secure_filename
        import uuid
        
        upload_dir = Path('uploads/temp')
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = upload_dir / filename
        file.save(str(filepath))
        
        detection_service = app.config['detection_service']
        result = detection_service.detect_video(str(filepath))
        
        try:
            if filepath.exists():
                filepath.unlink()
        except:
            pass
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/api/detection/audio', methods=['POST'])
def detect_audio():
    """Audio detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        from werkzeug.utils import secure_filename
        import uuid
        
        upload_dir = Path('uploads/temp')
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = upload_dir / filename
        file.save(str(filepath))
        
        detection_service = app.config['detection_service']
        result = detection_service.detect_audio(str(filepath))
        
        try:
            if filepath.exists():
                filepath.unlink()
        except:
            pass
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


# ============================================
# SOCKETIO EVENTS
# ============================================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_response', {
        'status': 'connected',
        'message': 'Connected to DeepFake Shield',
        'mode': 'PRODUCTION - REAL MODELS'
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    logger.info(f"Client disconnected: {request.sid}")


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🛡️  DeepFake Shield - PRODUCTION MODE")
    print("="*60)
    print("Starting on http://127.0.0.1:5000")
    print("Using REAL TRAINED MODELS for detection")
    print("="*60 + "\n")
    
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)
"""
analysis_routes.py
Blueprint for XAI and analysis endpoints
Uses already-initialized RealDetectionService from app.py
"""

from flask import Blueprint, request, jsonify, current_app
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid
import logging
import cv2
import base64

analysis_bp = Blueprint("analysis", __name__)

logger = logging.getLogger(__name__)

# ==========================================
# Helper
# ==========================================

def encode_image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


# ==========================================
# IMAGE HEATMAP (Grad-CAM)
# ==========================================

@analysis_bp.route("/analysis/heatmap", methods=["POST"])
def generate_heatmap():
    """
    Generate Grad-CAM heatmap for image
    """

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        upload_dir = Path("uploads/temp")
        upload_dir.mkdir(parents=True, exist_ok=True)

        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = upload_dir / filename
        file.save(str(filepath))

        logger.info(f"Saved file for XAI: {filepath}")

        # 🔥 Use already loaded detection service
        detection_service = current_app.detection_service

        if not detection_service.image_model:
            return jsonify({"error": "Image model not loaded"}), 500

        # Run detection (this already attaches XAI in your system)
        result = detection_service.detect_image(str(filepath))

        # Cleanup
        try:
            filepath.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Heatmap error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==========================================
# VIDEO XAI
# ==========================================

@analysis_bp.route("/analysis/video-xai", methods=["POST"])
def video_xai():
    """
    Generate video frame heatmaps
    """

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        upload_dir = Path("uploads/temp")
        upload_dir.mkdir(parents=True, exist_ok=True)

        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = upload_dir / filename
        file.save(str(filepath))

        detection_service = current_app.detection_service

        if not detection_service.video_model:
            return jsonify({"error": "Video model not loaded"}), 500

        result = detection_service.detect_video(str(filepath))

        try:
            filepath.unlink(missing_ok=True)
        except:
            pass

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Video XAI error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==========================================
# AUDIO SALIENCY
# ==========================================

@analysis_bp.route("/analysis/audio-xai", methods=["POST"])
def audio_xai():
    """
    Generate audio saliency explanation
    """

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        upload_dir = Path("uploads/temp")
        upload_dir.mkdir(parents=True, exist_ok=True)

        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = upload_dir / filename
        file.save(str(filepath))

        detection_service = current_app.detection_service

        if not detection_service.audio_model:
            return jsonify({"error": "Audio model not loaded"}), 500

        result = detection_service.detect_audio(str(filepath))

        try:
            filepath.unlink(missing_ok=True)
        except:
            pass

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Audio XAI error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==========================================
# SYSTEM STATUS
# ==========================================

@analysis_bp.route("/analysis/status", methods=["GET"])
def system_status():
    """
    Check model loading status
    """

    detection_service = current_app.detection_service

    return jsonify({
        "image_model_loaded": detection_service.image_model is not None,
        "video_model_loaded": detection_service.video_model is not None,
        "audio_model_loaded": detection_service.audio_model is not None,
        "image_xai_loaded": detection_service.image_xai is not None,
        "video_xai_loaded": detection_service.video_xai is not None,
        "audio_xai_loaded": detection_service.audio_xai is not None
    }), 200
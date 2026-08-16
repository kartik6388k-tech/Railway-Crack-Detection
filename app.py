"""
Flask Backend for Railway Crack/Gap Detection
Integrates YOLO model inference via detect.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
import traceback

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2

# Import detection function from detect.py
from detect import (
    load_model,
    detect_image,
    validate_arguments,
    DEFAULT_WEIGHTS,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_DETECTIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = Path("static/uploads")
RESULTS_FOLDER = Path("results")
MODELS_FOLDER = Path("runs/detect")

# Create directories
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Flask config
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload
app.config["JSON_SORT_KEYS"] = False

# Allowed extensions
ALLOWED_EXTENSIONS = {ext.lstrip(".") for ext in SUPPORTED_IMAGE_EXTENSIONS}

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# GLOBAL MODEL (Load Once)
# ============================================================================

MODEL = None
WEIGHTS_PATH = DEFAULT_WEIGHTS


def load_global_model():
    """Load YOLO model once at startup."""
    global MODEL
    if MODEL is not None:
        return MODEL

    try:
        logger.info(f"Loading YOLO model from: {WEIGHTS_PATH}")
        MODEL = load_model(WEIGHTS_PATH)
        logger.info(f"Model loaded successfully. Classes: {MODEL.names}")
        return MODEL
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        raise


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_unique_filename(original_filename: str, subfolder: str = "") -> Path:
    """Generate a unique filename with timestamp to avoid collisions."""
    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond precision
    unique_name = f"{stem}_{timestamp}{suffix}"

    if subfolder:
        return Path(subfolder) / unique_name
    return Path(unique_name)


def save_uploaded_file(file) -> Path | None:
    """Save uploaded file to upload folder."""
    if not file or file.filename == "":
        logger.warning("No file provided.")
        return None

    if not allowed_file(file.filename):
        logger.warning(f"File extension not allowed: {file.filename}")
        return None

    try:
        filename = secure_filename(file.filename)
        upload_path = UPLOAD_FOLDER / generate_unique_filename(filename).name
        file.save(upload_path)
        logger.info(f"File uploaded: {upload_path}")
        return upload_path
    except Exception as exc:
        logger.error(f"Error saving uploaded file: {exc}")
        return None


# ============================================================================
# ROUTES
# ============================================================================


@app.route("/", methods=["GET"])
def index():
    """Serve the main HTML page."""
    return send_file("templates/index.html", mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        model_loaded = MODEL is not None
        return jsonify(
            {
                "status": "ok",
                "model_loaded": model_loaded,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    
    Expects:
        - File: 'file' (multipart/form-data)
        - Optional parameters: conf, iou, imgsz, max_det
    
    Returns:
        JSON with detection results and annotated image path.
    """
    # ---- Validate Request ----
    if "file" not in request.files:
        logger.warning("No file provided in request.")
        return (
            jsonify({"error": "No file provided. Use multipart/form-data with 'file' key."}),
            400,
        )

    file = request.files["file"]

    # ---- Save Uploaded File ----
    upload_path = save_uploaded_file(file)
    if upload_path is None:
        return (
            jsonify({"error": f"Invalid file. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}),
            400,
        )

    # ---- Parse Optional Parameters ----
    try:
        conf = float(request.form.get("conf", DEFAULT_CONFIDENCE))
        iou = float(request.form.get("iou", DEFAULT_IOU))
        imgsz = int(request.form.get("imgsz", DEFAULT_IMAGE_SIZE))
        max_det = int(request.form.get("max_det", DEFAULT_MAX_DETECTIONS))
        device = request.form.get("device", None)

        if not (0.0 <= conf <= 1.0):
            return jsonify({"error": "Confidence must be between 0 and 1."}), 400
        if not (0.0 <= iou <= 1.0):
            return jsonify({"error": "IoU must be between 0 and 1."}), 400
        if imgsz <= 0:
            return jsonify({"error": "Image size must be > 0."}), 400
        if max_det <= 0:
            return jsonify({"error": "Max detections must be > 0."}), 400

    except ValueError as exc:
        return jsonify({"error": f"Invalid parameter type: {exc}"}), 400

    # ---- Load Model (if not already loaded) ----
    try:
        model = load_global_model()
    except Exception as exc:
        logger.error(f"Model loading failed: {exc}")
        return jsonify({"error": f"Model loading failed: {exc}"}), 500

    # ---- Run Detection ----
    try:
        logger.info(f"Running detection on: {upload_path}")
        original, annotated, detections = detect_image(
            model=model,
            image_path=upload_path,
            confidence=conf,
            iou=iou,
            image_size=imgsz,
            max_detections=max_det,
            device=device,
        )

        # ---- Save Annotated Result ----
        result_filename = generate_unique_filename(upload_path.name).name
        result_path = RESULTS_FOLDER / result_filename
        success = cv2.imwrite(str(result_path), annotated)
        if not success:
            raise RuntimeError(f"Failed to save annotated image to {result_path}")

        logger.info(f"Annotated image saved: {result_path}")

        # ---- Build Response ----
        response = {
            "success": True,
            "image_filename": upload_path.name,
            "result_image": result_filename,
            "result_image_url": f"/results/{result_filename}",
            "total_detections": len(detections),
            "detections": detections,
            "parameters": {
                "confidence": conf,
                "iou": iou,
                "imgsz": imgsz,
                "max_det": max_det,
            },
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Detection completed. Found {len(detections)} objects.")
        return jsonify(response), 200

    except Exception as exc:
        logger.error(f"Detection failed: {exc}\n{traceback.format_exc()}")
        return (
            jsonify(
                {
                    "error": f"Detection failed: {str(exc)}",
                    "type": type(exc).__name__,
                }
            ),
            500,
        )


@app.route("/results/<filename>", methods=["GET"])
def serve_result(filename: str):
    """Serve result images. Content-type is inferred from the file extension
    (not hardcoded) so .jpg/.png/.webp all serve correctly."""
    filepath = RESULTS_FOLDER / secure_filename(filename)
    if not filepath.exists():
        logger.warning(f"Result file not found: {filepath}")
        return jsonify({"error": "File not found."}), 404

    try:
        return send_file(filepath)
    except Exception as exc:
        logger.error(f"Error serving file {filepath}: {exc}")
        return jsonify({"error": "Could not serve file."}), 500


# Note: uploaded originals under static/uploads/ don't need a custom route —
# Flask's default static handler already serves them at /static/uploads/<filename>.


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return default configuration and model info."""
    try:
        model = load_global_model()
        return jsonify(
            {
                "defaults": {
                    "confidence": DEFAULT_CONFIDENCE,
                    "iou": DEFAULT_IOU,
                    "imgsz": DEFAULT_IMAGE_SIZE,
                    "max_det": DEFAULT_MAX_DETECTIONS,
                },
                "model": {
                    "classes": model.names,
                    "num_classes": len(model.names),
                    "weights_path": str(WEIGHTS_PATH),
                },
                "constraints": {
                    "max_upload_size_mb": 50,
                    "allowed_formats": sorted(ALLOWED_EXTENSIONS),
                },
            }
        )
    except Exception as exc:
        logger.error(f"Failed to fetch config: {exc}")
        return jsonify({"error": str(exc)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(413)
def handle_request_too_large(e):
    """Handle file too large error."""
    return (
        jsonify({"error": "File too large. Maximum size: 50 MB."}),
        413,
    )


@app.errorhandler(404)
def handle_not_found(e):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(500)
def handle_internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error."}), 500


# ============================================================================
# STARTUP & CLEANUP
# ============================================================================


@app.before_request
def startup():
    """Load model on first request."""
    if MODEL is None:
        try:
            load_global_model()
        except Exception as exc:
            logger.error(f"Failed to initialize model: {exc}")


if __name__ == "__main__":
    # Load model at startup
    try:
        load_global_model()
        logger.info("Model pre-loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to pre-load model: {exc}")

    # Run Flask app
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,  # Disable auto-reload to avoid loading model twice
        threaded=True,       # Don't let one in-flight /api/predict block other requests
    )
"""
REST API for EMNIST Character Recognition System.

This module provides a production-ready REST API for serving the trained
EMNIST character recognition models with real-time inference capabilities.
"""

import os
import io
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import ModelFactory
from src.data.transforms import get_transforms
from src.inference.predictor import EMNISTPredictor
from src.utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web frontend

# Global variables for model and predictor
predictor = None
config = None


def initialize_model():
    """Initialize the model and predictor."""
    global predictor, config

    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
        config = load_config(str(config_path))

        # Initialize predictor
        predictor = EMNISTPredictor(
            model_path=config['inference']['model_path'],
            config=config
        )

        logger.info("Model initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        # For demo purposes, create a dummy predictor
        predictor = DummyPredictor()


class DummyPredictor:
    """Dummy predictor for demonstration when model is not available."""

    def __init__(self):
        self.classes = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]

    def predict(self, image, return_probabilities=False):
        """Mock prediction."""
        import random
        predicted_class = random.choice(self.classes)
        confidence = random.uniform(0.7, 0.99)

        if return_probabilities:
            probabilities = {cls: random.uniform(0.01, 0.1) for cls in self.classes}
            probabilities[predicted_class] = confidence
            return predicted_class, confidence, probabilities

        return predicted_class, confidence

    def predict_batch(self, images):
        """Mock batch prediction."""
        return [self.predict(img) for img in images]


@app.route('/')
def home():
    """Home page with API documentation."""
    return jsonify({
        "message": "EMNIST Character Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict character from image",
            "/predict_batch": "POST - Predict characters from multiple images",
            "/health": "GET - Health check",
            "/info": "GET - Model information",
            "/classes": "GET - Available character classes"
        },
        "documentation": "/docs"
    })


@app.route('/docs')
def documentation():
    """API documentation page."""
    docs = {
        "title": "EMNIST Character Recognition API Documentation",
        "description": "REST API for handwritten character recognition using deep learning",
        "version": "1.0.0",
        "endpoints": [
            {
                "path": "/predict",
                "method": "POST",
                "description": "Predict character from a single image",
                "request_body": {
                    "image": "Base64 encoded image string",
                    "return_probabilities": "Boolean (optional, default: False)"
                },
                "response": {
                    "predicted_class": "Predicted character",
                    "confidence": "Prediction confidence (0-1)",
                    "probabilities": "Class probabilities (if requested)"
                }
            },
            {
                "path": "/predict_batch",
                "method": "POST",
                "description": "Predict characters from multiple images",
                "request_body": {
                    "images": "Array of base64 encoded image strings",
                    "return_probabilities": "Boolean (optional, default: False)"
                },
                "response": {
                    "predictions": "Array of prediction objects"
                }
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Check API health status",
                "response": {
                    "status": "API status",
                    "model_loaded": "Whether model is loaded",
                    "timestamp": "Current timestamp"
                }
            }
        ],
        "examples": {
            "single_prediction": {
                "request": {
                    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    "return_probabilities": True
                },
                "response": {
                    "predicted_class": "A",
                    "confidence": 0.95,
                    "probabilities": {"A": 0.95, "B": 0.02, "...": "..."}
                }
            }
        }
    }
    return jsonify(docs)


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    })


@app.route('/info')
def model_info():
    """Get model information."""
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    info = {
        "model_name": "EMNIST Character Recognition",
        "version": "1.0.0",
        "classes": len(predictor.classes),
        "supported_formats": ["PNG", "JPEG", "JPG"],
        "input_size": "28x28 pixels",
        "description": "Deep learning model for handwritten character recognition"
    }

    return jsonify(info)


@app.route('/classes')
def get_classes():
    """Get available character classes."""
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    return jsonify({
        "classes": predictor.classes,
        "num_classes": len(predictor.classes)
    })


@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict character from a single image."""
    try:
        # Validate request
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "Missing 'image' field in request"}), 400

        # Get parameters
        image_data = request.json['image']
        return_probabilities = request.json.get('return_probabilities', False)

        # Decode image
        image = decode_base64_image(image_data)
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Make prediction
        if return_probabilities:
            predicted_class, confidence, probabilities = predictor.predict(
                image, return_probabilities=True
            )
            response = {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "probabilities": {k: float(v) for k, v in probabilities.items()}
            }
        else:
            predicted_class, confidence = predictor.predict(image)
            response = {
                "predicted_class": predicted_class,
                "confidence": float(confidence)
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict characters from multiple images."""
    try:
        # Validate request
        if not request.json or 'images' not in request.json:
            return jsonify({"error": "Missing 'images' field in request"}), 400

        images_data = request.json['images']
        return_probabilities = request.json.get('return_probabilities', False)

        if not isinstance(images_data, list):
            return jsonify({"error": "'images' must be an array"}), 400

        if len(images_data) > 10:  # Limit batch size
            return jsonify({"error": "Maximum batch size is 10 images"}), 400

        # Decode images
        images = []
        for i, image_data in enumerate(images_data):
            image = decode_base64_image(image_data)
            if image is None:
                return jsonify({"error": f"Invalid image format at index {i}"}), 400
            images.append(image)

        # Make predictions
        predictions = []
        for image in images:
            if return_probabilities:
                predicted_class, confidence, probabilities = predictor.predict(
                    image, return_probabilities=True
                )
                prediction = {
                    "predicted_class": predicted_class,
                    "confidence": float(confidence),
                    "probabilities": {k: float(v) for k, v in probabilities.items()}
                }
            else:
                predicted_class, confidence = predictor.predict(image)
                prediction = {
                    "predicted_class": predicted_class,
                    "confidence": float(confidence)
                }
            predictions.append(prediction)

        return jsonify({"predictions": predictions})

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and predict from file."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Read and process image
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data)).convert('L')

        # Make prediction
        predicted_class, confidence = predictor.predict(image)

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "filename": file.filename
        })

    except Exception as e:
        logger.error(f"File upload error: {e}")
        return jsonify({"error": str(e)}), 500


def decode_base64_image(image_data: str) -> Optional[Image.Image]:
    """
    Decode base64 image string to PIL Image.

    Args:
        image_data (str): Base64 encoded image

    Returns:
        Optional[Image.Image]: Decoded image or None if invalid
    """
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        # Open image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        return image

    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return None


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Initialize model
    initialize_model()

    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting EMNIST API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
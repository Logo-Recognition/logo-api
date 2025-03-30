import base64
import os
import keras.saving
from flask import Blueprint, make_response, request, jsonify, send_from_directory
import io
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
from ultralytics import YOLO, RTDETR
from util import Util
import cv2
import keras
import gc

model_routes = Blueprint('model_routes', __name__)

AVAILABLE_DETECTION_MODEL = ['yolov8', 'yolov10', 'rtdetr']
AVAILABLE_CLASSIFICATION_MODEL = ['efficientnet', 'convnext', 'mobilenet', 'custom']

TEMP_DIR = 'temp_images'
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@model_routes.route('/api/model/run-realtime', methods=['POST'])
def run_realtime():
    """Processes images in real-time using detection and classification models."""
    
    detection_model_req = request.form.get('detection_model')
    classification_model_req = request.form.get('classification_model')

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Load detection model (Lazy loading, avoid keeping in memory for long)
    detection_model_map = {
        'yolov8': 'model/YOLOv8_47_1_class.pt',
        'yolov10': 'model/YOLOv10_47_1_class.pt',
        'rtdetr': 'model/RTDETR_47_1_class.pt'
    }
    detection_model_path = detection_model_map.get(detection_model_req)
    if not detection_model_path:
        return jsonify({'error': 'Invalid detection model selected'}), 400
    
    detection_model = (YOLO if "yolo" in detection_model_req else RTDETR)(detection_model_path).to(device)

    # Load classification model
    classification_model_map = {
        'efficientnet': ('model/EfficientNetV2_47.keras', 224),
        'convnext': ('model/Conv_47.keras', 224),
        'mobilenet': ('model/MobileNetV3_47.keras', 224),
        'custom': ('model/Custom_47.keras', 128)
    }
    classification_model_info = classification_model_map.get(classification_model_req)
    if not classification_model_info:
        return jsonify({'error': 'Invalid classification model selected'}), 400

    classification_model_path, IMG_SIZE = classification_model_info
    classification_model = keras.saving.load_model(classification_model_path)

    # Process images
    image_files = request.files.getlist("images")
    processed_image_urls = []

    for image_file in image_files:
        image_data = image_file.read()
        
        with Image.open(io.BytesIO(image_data)) as image:
            image_np = np.array(image.convert('RGB'))  # Efficient conversion

        # Run detection model
        predictions = detection_model(image_np)

        # Process the detected bounding boxes with classification
        processed_image_data = Util.draw_bounding_boxes(image_np, predictions, classification_model, IMG_SIZE)

        # Save the processed image to a temporary file
        file_name, file_extension = os.path.splitext(image_file.filename)
        temp_file_name = f"{file_name}_{detection_model_req}_{classification_model_req}_predicted.jpg"
        temp_file_path = os.path.join(TEMP_DIR, temp_file_name)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(processed_image_data)

        # Generate and store the image URL
        base_url = request.base_url.rsplit('/', 1)[0] 
        processed_image_urls.append({'predicted_url': f"{base_url}/{TEMP_DIR}/{temp_file_name}"})

        # Clear GPU cache after processing each image
        torch.cuda.empty_cache()

    # Cleanup to free memory
    del image_files, detection_model, classification_model, image_np, processed_image_data
    gc.collect()
    torch.cuda.empty_cache()

    return jsonify(processed_image_urls), 200

@model_routes.route('/api/model/temp_images/<filename>')
def get_temp_image(filename):
    """Route to serve temporary images."""
    return send_from_directory(TEMP_DIR, filename)
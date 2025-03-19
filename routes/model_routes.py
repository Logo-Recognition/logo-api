import base64
import os
import keras.saving
from flask import Blueprint, make_response, request, jsonify, g,send_from_directory
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib
matplotlib.use('Agg')
import torch
from ultralytics import YOLO,RTDETR
from util import Util
import cv2
import keras

model_routes = Blueprint('model_routes', __name__, )
AVAILABLE_DETECTION_MODEL=['yolov8','yolov10', 'rtdetr']
AVAILABLE_CLASSIFICATION_MODEL = ['efficientnet', 'convnext', 'mobilenet', 'custom']

TEMP_DIR = 'temp_images'
@model_routes.route('/api/model/run-realtime', methods=['POST'])
def run_realtime():
    detection_model_req = request.form.get('detection_model')
    classification_model_req = request.form.get('classification_model')

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    if detection_model_req == 'yolov8':
        detection_model = YOLO('model/YOLOv8_47_1_class.pt').to(device)
    elif detection_model_req == 'yolov10':
        detection_model = YOLO('model/YOLOv10_47_1_class.pt').to(device)
    elif detection_model_req == 'rtdetr':
        detection_model = RTDETR('model/RTDETR_47_1_class.pt').to(device)
    else:
        return jsonify({'error': 'Invalid detection model selected'}), 400

    if classification_model_req == 'efficientnet' :
        classification_model = keras.saving.load_model('model/EfficientNetV2_47.keras')
        IMG_SIZE = 224
    elif classification_model_req == 'convnext' :
        classification_model = keras.saving.load_model('model/Conv_47.keras')
        IMG_SIZE = 224
    elif classification_model_req == 'mobilenet' :
        classification_model = keras.saving.load_model('model/MobileNetV3_47.keras')
        IMG_SIZE = 224
    elif classification_model_req == 'custom' :
        classification_model = keras.saving.load_model('model/Custom_47.keras')
        IMG_SIZE = 128
    else:
        return jsonify({'error': 'Invalid classifiaciton model selected'}), 400

    # Get the image files from the request
    image_files = request.files.getlist("images")

    # Create the temporary directory if it doesn't exist
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # Create a list to store the processed image URLs
    processed_image_urls = []

    # Iterate over the image files
    for image_file in image_files:
        image_data = image_file.read()
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))

        # Use the model to predict the bounding boxes and class names
        predictions = detection_model(image_np)
        processed_image_data = Util.draw_bounding_boxes(image_np, predictions,classification_model,IMG_SIZE)
        
        # Save the processed image to a temporary file with "_predicted" suffix
        file_name, file_extension = os.path.splitext(image_file.filename)
        temp_file_name = f"{file_name}_{detection_model_req}_{classification_model_req}_predicted.jpg"
        temp_file_path = os.path.join(TEMP_DIR, temp_file_name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(processed_image_data)

        base_url = request.base_url.rsplit('/', 1)[0] 
        file_url = f"{base_url}/{TEMP_DIR}/{temp_file_name}"
        
        processed_image_urls.append({
            'predicted_url': file_url,
        })

    # Return the list of processed image URLs
    return jsonify(processed_image_urls), 200

@model_routes.route('/api/model/temp_images/<filename>')
def get_temp_image(filename):
    """Route to serve temporary images."""
    return send_from_directory(TEMP_DIR, filename)




    
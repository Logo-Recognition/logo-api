import io
import easyocr
from flask import Blueprint, g, request, jsonify, send_from_directory,current_app
import cv2
import numpy as np
import os
from PIL import Image
from util import Util


ocr_routes = Blueprint('ocr_routes', __name__)
TEMP_DIR = 'temp_images'
@ocr_routes.route('/api/ocr', methods=['POST'])
def run_ocr():
    # reader = current_app.config['OCR_READER']
    reader = easyocr.Reader(['th','en'],download_enabled=False,model_storage_directory='model')
    images = request.files.getlist('images')
    results = []

    # Create the temporary directory if it doesn't exist
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    for image in images:
        image_data = image.read()
        detection_results = reader.readtext(image_data)
        text = ' '.join([result[1] for result in detection_results])

        # Convert the image to PIL format
        image_pil = io.BytesIO(image_data)
        image_pil = Image.open(image_pil)

        # Draw bounding boxes and overlay text on the image
        processed_image_data = Util.draw_boxes(image_pil, detection_results)

        # Save the processed image to a temporary file
        file_name, file_extension = os.path.splitext(image.filename)
        temp_file_name = f"{file_name}_ocr.jpg"
        temp_file_path = os.path.join(TEMP_DIR, temp_file_name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(processed_image_data)

        # Create a URL for the temporary file
        base_url = request.base_url.rsplit('/', 1)[0]  # Remove the last part of the URL (/ocr)
        file_url = f"{base_url}/ocr/{TEMP_DIR}/{temp_file_name}"

        # Append the result to the list
        results.append({
            'text': text,
            'image_url': file_url
        })

    # Return the results as a JSON response
    return jsonify(results), 200

@ocr_routes.route('/api/ocr/temp_images/<filename>')
def get_temp_image(filename):
    """Route to serve temporary images."""
    return send_from_directory(TEMP_DIR, filename)
from minio import Minio
from minio.error import InvalidResponseError
from flask import jsonify, request
import torch
import ipaddress
import re
import cv2
import os
from ultralytics import YOLO,RTDETR
from minio import S3Error
import numpy as np
from PIL import ImageDraw,ImageFont,Image
import io
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v3 import preprocess_input
import gc
class Util:

    def is_correct_label_format(label):
        "x1 y1 x2 y2"
        try:
            items = label.split()
            if len(items) != 4:
                return False
            for item in items:
                float(item)
            return True
        except ValueError:
            return False

    def convert_to_yolo_format(x1, y1, x2, y2, image_width, image_height):
        # Calculate center coordinates
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        
        # Calculate width and height
        width = x2 - x1
        height = y2 - y1
        
        # Normalize coordinates by image dimensions
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        
        return x_center, y_center, width, height

    def upload_object(client, filename, data, length, bucket_name):
        
        client.put_object(bucket_name, filename, data, length)
        return jsonify({'success': True, "minio": f"{filename} is successfully uploaded to bucket {bucket_name}."})

    def allowed_file(filename, allowed_extensions):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

    def list_images(minio, bucket_name):
        try:
            # List all objects in the bucket
            objects = minio.list_objects(bucket_name, recursive=True)

            object_data = []
            for obj in objects:
                # Generate presigned URL for each object
                url = Util.generate_presigned_url(minio,bucket_name, obj.object_name)
                if url:
                    object_data.append({
                        'image_name' : obj.object_name,
                        'url' : url
                    })
            return object_data

        except InvalidResponseError as err:
            print(f"Error listing objects: {err}")
            return jsonify(error="Failed to list objects"), 500

    def generate_presigned_url(minio:Minio, bucket_name, object_name):
        try:
            # Generate presigned URL for object

            # url = minio.presigned_get_object(bucket_name, object_name)
            url = f"minio/{bucket_name}/{object_name}"
            return url
        
        except InvalidResponseError as err:
            print(f"Error generating presigned URL for {object_name}: {err}")
            return 
        
    def is_valid_bucket_name(bucket_name):
        # Check length (3-20 characters)
        if len(bucket_name) < 3 or len(bucket_name) > 20:
            return False

        # Check if the bucket name contains only lowercase letters and numbers
        if not re.match(r'^[a-z0-9]+$', bucket_name):
            return False

        return True

    def augment_image(image, augmentation_method):
        if augmentation_method == 'flip_horizontal':
            augmented_image = cv2.flip(image, 1)
        elif augmentation_method == 'flip_vertical':
            augmented_image = cv2.flip(image, 0)
        # Add more augmentation methods as needed
        else:
            return None

        return augmented_image

    def rename_dupplicate_image_name(imagename):
        # Split the image name and extension
        name, ext = os.path.splitext(imagename)

        # Find the highest numerical suffix in the name
        pattern = r'\((\d+)\)$'
        match = re.search(pattern, name)
        if match:
            suffix = int(match.group(1))
            name = name[:match.start()]
        else:
            suffix = 0

        # Increment the numerical suffix
        suffix += 1

        # Reassemble the new image name with the incremented suffix
        new_imagename = f"{name}({suffix}){ext}"
        return new_imagename
    
    def draw_bounding_boxes(image, results, classification_model, size):
        """Draw bounding boxes and classify detected objects efficiently."""

        # Convert image to BGR format (OpenCV uses BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Load class names once
        with open('model/47classes_classname.txt', 'r') as file:
            class_names = file.read().splitlines()

        # Iterate over detections
        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])  # Convert coordinates to int

                # Crop and preprocess the image in-place
                preprocess_img = Util.preprocess_img(image_bgr[y1:y2, x1:x2], size)
                class_num = np.argmax(classification_model.predict(preprocess_img))
                class_name = class_names[class_num]

                # Draw bounding box and label
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_bgr, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 1)

        # Convert image to bytes (JPEG format)
        _, buffer = cv2.imencode('.jpg', image_bgr)
        image_data = buffer.tobytes()

        # Explicitly release memory
        del image_bgr, buffer
        gc.collect()

        return image_data
    
    def draw_boxes(image, bounds, color='green', width=3):
        draw = ImageDraw.Draw(image)

        for bound in bounds:
  
            p0, p1, p2, p3 = bound[0]
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)

        # Convert the PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        processed_image_data = img_byte_arr.getvalue()

        return processed_image_data
    
    def preprocess_img(image,size):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img_rgb, (size, size))
        precessed_img = np.expand_dims(resized_img, axis=0)
        return precessed_img


    



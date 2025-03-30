import io
import logging
import os
import time
import random
from flask import Blueprint, jsonify, request, send_from_directory
import keras
import numpy as np
import requests
import tensorflow as tf
import tweepy
import gc
from datetime import datetime, timedelta
from PIL import ImageDraw, Image
from ultralytics import YOLO, RTDETR
import cv2
from ultralytics.utils.plotting import Annotator

# Multiple API keys configuration
API_KEYS = [
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_MAIN"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_MAIN"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_MAIN"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_MAIN"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_MAIN"),
    },
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_ALT"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_ALT"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_ALT"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_ALT"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_ALT"),
    },
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_ALT_1"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_ALT_1"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_ALT_1"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_ALT_1"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_ALT_1"),
    },
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_ALT_2"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_ALT_2"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_ALT_2"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_ALT_2"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_ALT_2"),
    },
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_ALT_3"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_ALT_3"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_ALT_3"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_ALT_3"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_ALT_3"),
    },
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_ALT_4"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_ALT_4"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_ALT_4"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_ALT_4"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_ALT_4"),
    },
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_ALT_5"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_ALT_5"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_ALT_5"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_ALT_5"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_ALT_5"),
    },
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_ALT_6"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_ALT_6"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_ALT_6"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_ALT_6"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_ALT_6"),
    },
    {
        "BEARER_TOKEN": os.getenv("TWEEPY_BEARER_TOKEN_ALT_7"),
        "API_KEY": os.getenv("TWEEPY_API_KEY_ALT_7"),
        "API_SECRET": os.getenv("TWEEPY_API_SECRET_ALT_7"),
        "ACCESS_TOKEN": os.getenv("TWEEPY_ACCESS_TOKEN_ALT_7"),
        "ACCESS_TOKEN_SECRET": os.getenv("TWEEPY_ACCESS_TOKEN_SECRET_ALT_7"),
    },
]
# Track which API key is currently in use
current_api_key_index = 0
# Track when each API key was last rate limited
api_key_cooldowns = {i: 0 for i in range(len(API_KEYS))}

TEMP_DIR = 'temp_images'
AVAILABLE_DETECTION_MODEL=['yolov8','yolov10', 'rtdetr']
AVAILABLE_CLASSIFICATION_MODEL = ['efficientnet', 'convnext', 'mobilenet', 'custom']

# Initialize Tweepy clients and API handlers
clients = []
apis = []

# Global model caching
_detection_model = None
_classification_model = None
_classification_input_size = None
_class_names_list = None

def load_models_if_needed(detection_model_name, classification_model_name):
    """Load models only if they haven't been loaded already or if they've changed"""
    global _detection_model, _classification_model, _classification_input_size, _class_names_list
    
    # Load detection model if needed
    if _detection_model is None or not hasattr(_detection_model, '_name') or _detection_model._name != detection_model_name:
        # Clear previous model from memory if it exists
        if _detection_model is not None:
            del _detection_model
            gc.collect()  
            
        if detection_model_name == 'yolov8':
            _detection_model = YOLO('model/YOLOv8_47_1_class.pt')
            _detection_model._name = 'yolov8'  # Add name attribute for checking
        elif detection_model_name == 'yolov10':
            _detection_model = YOLO('model/YOLOv10_47_1_class.pt')
            _detection_model._name = 'yolov10'
        elif detection_model_name == 'rtdetr':
            _detection_model = RTDETR('model/RTDETR_47_1_class.pt')
            _detection_model._name = 'rtdetr'
    
    # Load classification model if needed
    if _classification_model is None or not hasattr(_classification_model, '_name') or _classification_model._name != classification_model_name:
        # Clear previous model from memory if it exists
        if _classification_model is not None:
            del _classification_model
            keras.backend.clear_session()  # Clear Keras session
            gc.collect()  # Force garbage collection
            
        if classification_model_name == 'efficientnet':
            _classification_model = keras.models.load_model('model/EfficientNetV2_47.keras')
            _classification_model._name = 'efficientnet'  # Add name attribute
            _classification_input_size = (224, 224)
        elif classification_model_name == 'convnext':
            _classification_model = keras.models.load_model('model/Conv_47.keras')
            _classification_model._name = 'convnext'
            _classification_input_size = (224, 224)
        elif classification_model_name == 'mobilenet':
            _classification_model = keras.models.load_model('model/MobileNetV3_47.keras')
            _classification_model._name = 'mobilenet'
            _classification_input_size = (224, 224)
        elif classification_model_name == 'custom':
            _classification_model = keras.models.load_model('model/Custom_47.keras')
            _classification_model._name = 'custom'
            _classification_input_size = (128, 128)
    
    # Load class names if not already loaded
    if _class_names_list is None:
        class_names_file = 'model/47classes_classname.txt'
        with open(class_names_file, 'r') as file:
            _class_names_list = [line.strip() for line in file.readlines()]
    
    return _detection_model, _classification_model, _classification_input_size, _class_names_list

def initialize_tweepy_clients():
    """Initialize all Tweepy clients and APIs from the available API keys"""
    global clients, apis
    clients = []
    apis = []
    
    for key_set in API_KEYS:
        if all(key_set.values()):  # Ensure all keys are present
            client = tweepy.Client(
                bearer_token=key_set["BEARER_TOKEN"],
                consumer_key=key_set["API_KEY"],
                consumer_secret=key_set["API_SECRET"],
                access_token=key_set["ACCESS_TOKEN"],
                access_token_secret=key_set["ACCESS_TOKEN_SECRET"]
            )
            
            auth = tweepy.OAuth1UserHandler(
                consumer_key=key_set["API_KEY"],
                consumer_secret=key_set["API_SECRET"],
                access_token=key_set["ACCESS_TOKEN"],
                access_token_secret=key_set["ACCESS_TOKEN_SECRET"]
            )
            
            api = tweepy.API(auth)
            
            clients.append(client)
            apis.append(api)
            
    # Log how many valid API keys were found
    logging.info(f"Initialized {len(clients)} Twitter API clients")
    
    if not clients:
        logging.error("No valid Twitter API credentials found!")

# Initialize on module load
initialize_tweepy_clients()

def get_next_available_api():
    """Get the next available API key that isn't on cooldown"""
    global current_api_key_index
    
    current_time = time.time()
    attempts = 0
    
    while attempts < len(API_KEYS):
        # Check if current API key is off cooldown
        if current_time > api_key_cooldowns[current_api_key_index]:
            return current_api_key_index
        
        # Move to next API key
        current_api_key_index = (current_api_key_index + 1) % len(API_KEYS)
        attempts += 1
    
    # If all API keys are on cooldown, return the one with the earliest cooldown expiry
    earliest_index = min(api_key_cooldowns, key=api_key_cooldowns.get)

    return earliest_index

scrape_routes = Blueprint('scrape_routes', __name__)
@scrape_routes.route('/api/scrape', methods=['GET'])
def search_tweets_with_images():
    if request.method != 'GET':
        return jsonify({"error": "Method not allowed"}), 405
    
    keyword = request.args.get("keyword", type=str)
    start_date = request.args.get(
        "start_date", (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), type=str
    )
    end_date = request.args.get(
        "end_date", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), type=str
    )
    detection_model_name = request.args.get(
        "detect_model", "yolov8", type=str
    )
    classification_model_name = request.args.get(
        "class_model", "custom", type=str
    )

    if detection_model_name not in AVAILABLE_DETECTION_MODEL:
        return jsonify({"error": f"Invalid detection model. Available models are: {', '.join(AVAILABLE_DETECTION_MODEL)}"}), 400

    if classification_model_name not in AVAILABLE_CLASSIFICATION_MODEL:
        return jsonify({"error": f"Invalid classification model. Available models are: {', '.join(AVAILABLE_CLASSIFICATION_MODEL)}"}), 400
    
    try:
        # Load models only if needed (using the new function)
        detection_model, classification_model, classification_input_size, class_names_list = load_models_if_needed(
            detection_model_name, classification_model_name
        )
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    

    start_time = f"{start_date}T00:00:00Z"
    end_time = f"{end_date}T23:59:59Z"
    query = f"{keyword} has:images -is:retweet"

    max_images = 6
    image_count = 0
    images_urls = []

    base_url = request.url_root.rstrip('/')

    # Maximum retry attempts across all API keys
    max_retries = len(API_KEYS) * 2
    retry_count = 0
    found_enough = False
    
    while retry_count < max_retries:
        try:
            # Get the next available API key
            api_index = get_next_available_api()
            client = clients[api_index]
            api = apis[api_index]
            logging.info(f"Using API key #{api_index + 1} for Twitter request")
            
            # Fetch tweets with images
            tweets = client.search_recent_tweets(
                query=query,
                start_time=start_time,
                end_time=end_time,
                max_results=10,
                expansions=["attachments.media_keys", "author_id"],
                media_fields=["url", "type"],
                tweet_fields=["id"]
            )

            # Check if media exists in response
            if not tweets.includes or "media" not in tweets.includes:
                return jsonify({"error": "No images found for the given keyword and date range."}), 400

            # Create a dictionary for media lookup
            media_dict = {media.media_key: media for media in tweets.includes["media"]}

            for tweet in tweets.data:
                if found_enough:
                    break
                if "attachments" not in tweet or "media_keys" not in tweet.attachments:
                    continue

                tweet_url = f"https://twitter.com/{tweet.author_id}/status/{tweet.id}"

                for media_key in tweet.attachments["media_keys"]:
                    media = media_dict.get(media_key)

                    # ✅ Process only photos
                    if media and media.type == "photo":
                        img_url = media.url
                        img_path = os.path.join(TEMP_DIR, img_url.split("/")[-1])

                        # Download image
                        img_data = requests.get(img_url).content
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                            print(f"Image saved: {img_path}")

                        image_count += 1
                        image_name = img_url.split("/")[-1]
                        image_file_path = f"{base_url}/api/scrape/{TEMP_DIR}/{image_name}"

                        # YOLO model inference
                        image = Image.open(img_path)
                        results = detection_model.predict(image)

                        if len(results[0].boxes) <= 0:
                            images_urls.append({"image_url": image_file_path, "tweet_url": tweet_url})
                        else :
                            class_names = []
                            annotator = Annotator(image)
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cropped_image = image.crop((x1, y1, x2, y2)).convert("RGB")
                                cropped_array = keras.utils.img_to_array(cropped_image)
                                input_tensor = tf.image.resize(cropped_array, classification_input_size)
                                input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
                                input_tensor = tf.expand_dims(input_tensor, axis=0)
                                class_prediction = classification_model.predict(input_tensor)
                                predicted_class = np.argmax(class_prediction)
                                class_name = class_names_list[predicted_class]
                                class_names.append(class_name)
                                annotator.box_label(box.xyxy[0], class_name)

                            # Convert NumPy array to PIL Image
                            annotated_img = Image.fromarray(annotator.result())

                            # Save annotated image
                            temp_file_name = f"annotated_{image_name}"
                            annotated_img_path = os.path.join(TEMP_DIR, temp_file_name)
                            annotated_img.save(annotated_img_path)

                            # Store annotated image URL
                            annotated_image_url = f"{base_url}/api/scrape/{TEMP_DIR}/{temp_file_name}"
                            images_urls.append({"image_url": annotated_image_url, "tweet_url": tweet_url, "class_names": class_names,})
                            print(f'image count : {image_count}')

                        del image
                        if 'annotated_img' in locals():
                            del annotated_img
                        del results

                        if image_count >= max_images:
                            found_enough=True
                            break

            keras.backend.clear_session()
            gc.collect()
            if image_count == 0:
                return jsonify({"error": "No images found in the retrieved tweets."}), 400

            return jsonify({"predicted_url": images_urls}), 200

        except tweepy.TooManyRequests as e:
            # Set cooldown for the current API key - properly parse the Unix timestamp
            reset_time = int(e.response.headers.get("x-rate-limit-reset", 0))
            # Store the actual timestamp, not a duration
            api_key_cooldowns[api_index] = reset_time
            
            logging.warning(f"Rate limit exceeded for API key #{api_index + 1}. Switching to next available key.")
            retry_count += 1
            
            # Check if all APIs are rate limited
            all_limited = all(cooldown > time.time() for cooldown in api_key_cooldowns.values())
            
            if all_limited:
                min_cooldown = min(api_key_cooldowns.values())
                wait_time = min_cooldown - time.time()
                # Don't sleep! Return an error to the client
                return jsonify({
                    "error": f"All Twitter API keys are rate limited. Please try again in {int(wait_time)} seconds."
                }), 429  # 429 is the HTTP status code for "Too Many Requests"

        except Exception as e:
            logging.error(f"Error with API key #{api_index + 1}: {str(e)}")
            retry_count += 1
            
            # Add a small delay before trying the next key
            time.sleep(2)
            
            # If we've tried all keys multiple times, give up
            if retry_count >= max_retries:
                return jsonify({"error": f"Failed after {max_retries} attempts: {str(e)}"}), 400

    # This should not be reached if the function returns properly above
    return jsonify({"error": "Failed to retrieve tweets after exhausting all API keys"}), 400


@scrape_routes.route('/api/scrape/temp_images/<filename>')
def get_temp_image(filename):
    """Route to serve temporary images."""
    return send_from_directory(TEMP_DIR, filename), 200
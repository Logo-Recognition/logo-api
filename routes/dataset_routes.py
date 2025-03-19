from io import BytesIO
from zipstream import ZipFile
import cv2
from flask import Blueprint, Response, request, jsonify, g
import numpy as np
from sklearn.model_selection import train_test_split
from db_service import DB_service
import imgaug.augmenters as iaa
import imgaug as ia  

dataset_routes = Blueprint('dataset_routes', __name__)

@dataset_routes.route('/api/dataset', methods=['GET'])
def get_dataset_summary():
    db_connection = g.db_connection
    try:
        total_images = len(DB_service.get_annotated_images(db_connection))
        total_classes = len(DB_service.get_classes_name(db_connection))
        return jsonify({
            'total_classes': total_classes,
            'total_images': total_images
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@dataset_routes.route('/api/dataset/download', methods=['POST'])
def download_zip():
    minio_client = g.minio_client
    db_connection = g.db_connection

    zip_data = ZipFile(mode='w') 
    images = []
    file_names = []
    train_test_split_params  = request.get_json()['train_test_split_param']
    train_size = train_test_split_params["train_size"]
    test_size = train_test_split_params["test_size"]
    valid_size = train_test_split_params["valid_size"]

    try:
        annotated = 'annotated'
        buckets = minio_client.list_buckets()
        for bucket in buckets :
            if bucket.name == annotated :
                for obj in minio_client.list_objects(bucket.name, recursive=True):                  
                        data = minio_client.get_object(bucket.name, obj.object_name).read()
                        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)    
                        # image.object_name = obj.object_name
                        file_names.append(obj.object_name)            
                        images.append(image)
                         
        # Get augmentation parameters from request
        augmentation_params  = request.get_json()["augmentation_param"]
         # Create augmenters based on the received parameters
        print(augmentation_params)
        augmenters = []
        if 'rotate' in augmentation_params and augmentation_params['rotate'] != 0 :
            rotate_angles = augmentation_params['rotate']
            augmenters.append(iaa.Rotate(rotate_angles))
            augmenters.append(iaa.Rotate(-rotate_angles))

        if 'flip_horizontal' in augmentation_params and augmentation_params['flip_horizontal']:
            #bool
            augmenters.append(iaa.Fliplr(1.0))

        if 'flip_verical' in augmentation_params and augmentation_params['flip_verical']:
             #bool
            augmenters.append(iaa.Flipud(1.0))

        if 'gaussian_noise' in augmentation_params and augmentation_params['gaussian_noise'] != 0:
            #range 0.0-255.0  use 0-0.1
            gaussian_noise_params = augmentation_params['gaussian_noise']
            augmenters.append(iaa.AdditiveGaussianNoise(scale = gaussian_noise_params*255,per_channel=True))
        
        if 'pepper_noise' in augmentation_params and augmentation_params['pepper_noise'] != 0:
             #range 0.0-255.0 use 0-0.1
            pepper_noise_params = augmentation_params['pepper_noise']
            augmenters.append(iaa.AdditiveGaussianNoise(scale = pepper_noise_params*255,per_channel=False))
        
        if 'scaling' in augmentation_params and augmentation_params['scaling'] != 1:
            #use 0.1-2 norm1
            scaling_params  = augmentation_params['scaling']
            augmenters.append(iaa.Affine(scale = scaling_params))

        if 'brightness' in augmentation_params and augmentation_params['brightness'] != 1:
            #range0-3 use 0.1-2 norm1
            brightness_params = augmentation_params['brightness']
            augmenters.append(iaa.MultiplyBrightness((brightness_params)))
        
        if 'saturation' in augmentation_params and augmentation_params['saturation'] != 1:
            #range0-None use0-2 norm1
            saturation_params = augmentation_params['saturation']
            augmenters.append(iaa.MultiplySaturation((saturation_params)))

        if 'contrast' in augmentation_params and augmentation_params['contrast'] != 1:
            #range0-None use0-2 norm1
            contrast_params = augmentation_params['contrast']
            augmenters.append(iaa.LinearContrast((contrast_params)))
        
        def augment_data(image, annotations, augmenter):
            yolo_annotations = ""
            if augmenter == None:
                for annote in annotations:
                    image_shape = image.shape[:2]
                    x1, y1, x2, y2, class_id = annote['x1'], annote['y1'], annote['x2'], annote['y2'],annote['cid']
                    x = (x1 + x2) / (2 * image_shape[1])  
                    y = (y1 + y2) / (2 * image_shape[0])  
                    w = (x2 - x1) / image_shape[1]  
                    h = (y2 - y1) / image_shape[0]  
                    yolo_annotation = f"{int(class_id)} {x} {y} {w} {h} \n"
                    yolo_annotations += yolo_annotation
                return image,yolo_annotations
                
            # Convert image and annotations to imgaug formats
            bbs = [ia.BoundingBox(annote['x1'], annote['y1'], annote['x2'], annote['y2'],annote['cid']) for annote in annotations]
            bb_on_img = ia.BoundingBoxesOnImage(bbs, shape=image.shape)

            image_aug, annote_aug = augmenter(image=image.copy(),bounding_boxes = bb_on_img.copy())
    
            for augmented_image, augmented_annotation in zip(image_aug, annote_aug):
                image_shape = augmented_image.shape[:2]
                x1, y1, x2, y2, class_id = augmented_annotation.x1,augmented_annotation.y1,augmented_annotation.x2,augmented_annotation.y2,augmented_annotation.label
                x = (x1 + x2) / (2 * image_shape[1])  
                y = (y1 + y2) / (2 * image_shape[0])  
                w = (x2 - x1) / image_shape[1]  
                h = (y2 - y1) / image_shape[0]  
                yolo_annotation = f"{int(class_id)} {x} {y} {w} {h} \n"
                yolo_annotations += yolo_annotation

            return image_aug, yolo_annotations

        augmented_names = []
        all_image = [] 
        all_annote = []

        for image, file_name in zip(images, file_names):
            image_name = file_name
            annote_data = DB_service.get_annote_by_imagename(db_connection, image_name)
            original_image,original_annote = augment_data(image, annote_data, None)
            all_annote.append(original_annote)
            augmented_names.append(image_name)
            all_image.append(original_image)
             # ตั้งชื่อไฟล์ใหม่สำหรับรูปภาพและไฟล์ annotation ที่ผ่านการ augment
            base_name = image_name.rsplit('.', 1)[0]
            for augmenter in augmenters:
                augment_type = augmenter.__class__.__name__.lower()
                augmented_images, yolo_annotations = augment_data(image, annote_data, augmenter)
                all_annote.append(yolo_annotations)
                all_image.append(augmented_images)

                if augment_type == 'rotate':
                    rotate_angle = augmenter.rotate.value
                    augmented_name = f"{base_name}_{augment_type}_{rotate_angle}.jpg"
                    augmented_names.append(augmented_name)
                elif augment_type == 'fliplr':
                    augmented_name = f"{base_name}_flip-horizontal.jpg"
                    augmented_names.append(augmented_name)
                elif augment_type == 'flipud':
                    augmented_name = f"{base_name}_flip-verical.jpg"
                    augmented_names.append(augmented_name)
                elif augment_type == 'additivegaussiannoise' and augmenter.per_channel.value == True:
                    augmented_name = f"{base_name}_gaussian-noise_{augmentation_params['gaussian_noise']}.jpg"
                    augmented_names.append(augmented_name)
                elif augment_type == 'additivegaussiannoise' and augmenter.per_channel.value == False:
                    augmented_name = f"{base_name}_salt-pepper-noise_{augmentation_params['pepper_noise']}.jpg"
                    augmented_names.append(augmented_name)
                elif augment_type == 'affine':
                    augmented_name = f"{base_name}_scaling_{scaling_params}.jpg"
                    augmented_names.append(augmented_name)
                elif augment_type == 'multiplybrightness':                 
                    augmented_name = f"{base_name}_brightness_{brightness_params}.jpg"
                    augmented_names.append(augmented_name)
                elif augment_type == 'multiplysaturation':                 
                    augmented_name = f"{base_name}_saturation_{saturation_params}.jpg"
                    augmented_names.append(augmented_name)
                elif augment_type == 'linearcontrast':                 
                    augmented_name = f"{base_name}_contrast_{contrast_params}.jpg"
                    augmented_names.append(augmented_name)
        
        X = all_image 
        y = all_annote
        names = augmented_names  

        def train_test_valid_split(X, y, file_names, train_size=0.6, test_size=0.2, valid_size=0.2, random_state=42):
            """
            Splits the data into train, test, and validation sets.
            
            Args:
                X (numpy.ndarray): Input features.
                y (numpy.ndarray): Target labels.
                file_names (list): List of file names corresponding to the input data.
                train_size (float, optional): Proportion of the dataset to include in the train split. Default is 0.6.
                test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
                valid_size (float, optional): Proportion of the dataset to include in the validation split. Default is 0.2.
                random_state (int, optional): Random state for reproducibility. Default is 42.
                
            Returns:
                tuple: Tuples of train, test, and validation data splits (X_train, X_test, X_valid, y_train, y_test, y_valid, train_names, test_names, valid_names).
            """
            
            # Check if the sum of sizes is greater than 1
            if train_size + test_size + valid_size > 1.0:
                raise ValueError("The sum of train_size, test_size, and valid_size must not exceed 1.")
            
            if test_size == 0 and valid_size == 0 or len(X) <= 1 :
                X_train, y_train, train_names = X, y, file_names
                X_test, y_test, test_names = np.array([]), np.array([]), []
                X_valid, y_valid, valid_names = np.array([]), np.array([]), []
                return X_train, X_test, X_valid, y_train, y_test, y_valid, train_names, test_names, valid_names

            # Calculate the remaining size
            remaining_size = 1.0 - train_size - test_size - valid_size
            
            # Split the data into train and temp
            X_train, X_temp, y_train, y_temp, train_names, temp_names = train_test_split(X, y, file_names, train_size=train_size / (1 - remaining_size), random_state=random_state)

            # Split the temp data into test and valid
            if len(X_temp) == 1:
                X_test, y_test, test_names = X_temp, y_temp, temp_names
                X_valid, y_valid, valid_names = np.array([]), np.array([]), []
                print('here')
            else:
                # Split the temp data into test and valid
                if test_size == 0:
                    X_test, y_test, test_names = np.array([]), np.array([]), []
                    X_valid, y_valid, valid_names = X_temp, y_temp, temp_names
                elif valid_size == 0:
                    X_valid, y_valid, valid_names = np.array([]), np.array([]), []
                    X_test, y_test, test_names = X_temp, y_temp, temp_names
                else:
                    test_size = test_size / (test_size + valid_size)
                    X_test, X_valid, y_test, y_valid, test_names, valid_names = train_test_split(X_temp, y_temp, temp_names, test_size=test_size, random_state=random_state)
            return X_train, X_test, X_valid, y_train, y_test, y_valid, train_names, test_names, valid_names
        
        X_train, X_test, X_valid, y_train, y_test, y_valid, names_train, names_test, names_valid = train_test_valid_split(X, y, names, train_size=train_size, test_size=test_size, valid_size=valid_size  )
        
  
        class_names = DB_service.get_classes_name(db_connection)
        class_id = []
        for i in class_names:
            class_id.append(DB_service.get_class_index_by_classname(db_connection, i))

        # Generate YAML content
        total_classes = len(class_names)
        class_names_yaml = ""

        if class_id is not None:
            for cid, class_name in zip(class_id, class_names):
                class_names_yaml += f"  {cid}: {class_name}\n"
        else:
            return jsonify({'error': "class_id is None"})

        yaml_content = f"""
# YAML file for YOLO dataset format

# Path to the dataset directory
path: /path/to/dataset

# Train, validation, and test splits
train: train
val: validate
test: test

# Number of classes
nc: {total_classes}

# Class names
names:
{class_names_yaml}
"""

        # Add the YAML file to the ZIP file
        zip_data.write_iter('yolo_dataset_format.yaml', BytesIO(yaml_content.encode('utf-8')))

        def add_images_to_zip(images, labels, names, folder_name):
            for image, label, name in zip(images, labels, names):
                _, img_encoded = cv2.imencode('.jpg', image)
                image_name = f"{name}"
                label_name = f"{name.rsplit('.', 1)[0]}.txt"
                zip_data.write_iter(f'{folder_name}/{image_name}', BytesIO(img_encoded.tobytes()))
                zip_data.write_iter(f'{folder_name}/{label_name}', BytesIO(label.encode('utf-8')))
        
        add_images_to_zip(X_train, y_train ,names_train, 'train')
        add_images_to_zip(X_valid, y_valid,names_valid, 'validate')
        add_images_to_zip(X_test, y_test,names_test, 'test')

        def zip_generator():
            for data in zip_data:
                yield data

        return Response(
            zip_generator(),
            mimetype='application/zip',
            headers={'Content-Disposition': 'attachment; filename=images.zip'}
        )

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
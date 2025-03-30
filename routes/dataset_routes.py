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
        
        def convert_pixel_to_yolo(image_shape, x1, y1, x2, y2):
            """
            Convert pixel coordinates to YOLO format
            """
            img_height, img_width = image_shape[:2]
            x_center = (x1 + x2) / (2 * img_width)
            y_center = (y1 + y2) / (2 * img_height)
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            return x_center, y_center, width, height

        def augment_data(image, annotations, augmenter):
            """
            Augment image and bounding boxes
            """
            # Convert annotations to imgaug bounding boxes
            bbs = []
            for annote in annotations:
                # Convert YOLO format to pixel coordinates
                img_height, img_width = image.shape[:2]
                x1, y1, x2, y2, cid = annote['x1'], annote['y1'], annote['x2'], annote['y2'],annote['cid']
                bbs.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=cid))
            
            # Create BoundingBoxesOnImage object
            bb_on_img = ia.BoundingBoxesOnImage(bbs, shape=image.shape)

            # Apply augmentation
            if augmenter is not None:
                image_aug, bb_aug = augmenter(image=image.copy(), bounding_boxes=bb_on_img.copy())
            else:
                image_aug, bb_aug = image, bb_on_img

            # Convert augmented bounding boxes back to YOLO format
            yolo_annotations = ""
            for aug_bb in bb_aug.bounding_boxes:
                # Convert pixel coordinates back to YOLO format
                x_center, y_center, width, height = convert_pixel_to_yolo(
                    image_aug.shape, 
                    aug_bb.x1, aug_bb.y1, aug_bb.x2, aug_bb.y2
                )
                
                # Create YOLO annotation string
                yolo_annotation = f"{aug_bb.label} {x_center} {y_center} {width} {height}\n"
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
            Split data into training, testing, and validation sets.
            
            Parameters:
            -----------
            X : array-like
                Features to split
            y : array-like
                Target values to split
            file_names : array-like
                File names corresponding to each sample
            train_size : float, default=0.6
                Proportion of data for training set
            test_size : float, default=0.2
                Proportion of data for test set
            valid_size : float, default=0.2
                Proportion of data for validation set
            random_state : int, default=42
                Random seed for reproducibility
                
            Returns:
            --------
            X_train, X_test, X_valid : array-like
                Split feature sets
            y_train, y_test, y_valid : array-like
                Split target sets
            train_names, test_names, valid_names : array-like
                Split file names
            """
            from sklearn.model_selection import train_test_split
            import numpy as np
            
            # Check that proportions sum to 1
            if abs(train_size + test_size + valid_size - 1.0) > 1e-10:
                raise ValueError("train_size, test_size, and valid_size must sum to 1.0")
            
            total_samples = len(X)
            
            if total_samples == 0:
                return (np.array([]), np.array([]), np.array([]), 
                        np.array([]), np.array([]), np.array([]),
                        [], [], [])
            
            # Case where both test and validation are 0
            if test_size == 0 and valid_size == 0:
                # All data goes to training
                return (X, np.array([]), np.array([]),
                        y, np.array([]), np.array([]),
                        file_names, [], [])
            
            # First split: separate training set
            X_train, X_temp, y_train, y_temp, train_names, temp_names = train_test_split(
                X, y, file_names, test_size=(test_size + valid_size), 
                train_size=train_size, random_state=random_state, shuffle=True
            )
            
            # If either test or validation is zero-sized
            if test_size == 0:
                return (X_train, np.array([]), X_temp, 
                        y_train, np.array([]), y_temp,
                        train_names, [], temp_names)
            
            if valid_size == 0:
                return (X_train, X_temp, np.array([]), 
                        y_train, y_temp, np.array([]),
                        train_names, temp_names, [])
            
            # Second split: divide the remaining data into test and validation
            # Calculate relative sizes for the second split
            relative_test_size = test_size / (test_size + valid_size)
            
            X_test, X_valid, y_test, y_valid, test_names, valid_names = train_test_split(
                X_temp, y_temp, temp_names, test_size=(1 - relative_test_size),
                train_size=relative_test_size, random_state=random_state, shuffle=True
            )
            
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
        return jsonify({'error': str(e)}), 500
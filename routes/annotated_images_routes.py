from io import BytesIO
import cv2
from flask import Blueprint, request, jsonify, g
from minio import S3Error
import numpy as np
from util import Util
from db_service import DB_service
from minio.commonconfig import CopySource

annotated_images_routes = Blueprint('annotated_images_routes', __name__)

@annotated_images_routes.route('/api/annotated-images', methods=['GET','POST'])
def manage_annotated_images():
    db_connection = g.db_connection
    minio_client = g.minio_client
    
    if request.method == 'GET':
        try : 
            annotated_images = DB_service.get_annotated_images(db_connection)
            for annotated_image in annotated_images :
                annotated_image['image_name'] = annotated_image['image']
                annotated_image['image'] = Util.generate_presigned_url(minio_client,'annotated',annotated_image['image'])
            return jsonify(annotated_images)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    elif request.method == 'POST':
        json = request.get_json()
        image_name = json.get('image_name')
        print(image_name)
        label_list = json.get('label')
        try:
            if label_list == [] or label_list == None :
                if not DB_service.is_image_exist(db_connection,image_name):
                    return jsonify({'success': False,'error': 'Label should not be empty for create annote.'}), 402
                if not DB_service.delete_all_annote_by_imagename(db_connection,image_name):
                    return jsonify({'success': False,'error': f'Error when delete exist annote.'}), 402
                if not DB_service.delete_image_by_id(db_connection, DB_service.get_id_by_imagename(db_connection,image_name)):
                    return jsonify({'success': False,'error': f'Error when delete exist image.'}), 402
                minio_client.copy_object('unannotated',image_name, CopySource('annotated',image_name))
                minio_client.remove_object('annotated',image_name)
                return jsonify({'success': True, 'message': f'Annotated image has deleted.'}), 201

            
            for label in label_list :
                class_name = label['class_name']
                bbox = label['bbox']
                #เช้คว่ามี class ใน db มั้ย
                if class_name not in DB_service.get_classes_name(db_connection):
                    return jsonify({'success': False,'error': f'Not found class name : {class_name}.'}), 402
                #เช้คว่ามี label ที่ผิด format มั้ย
                if not Util.is_correct_label_format(bbox):
                    return jsonify({'success': False,'error': f'Label {class_name} format is not correct.'}), 402
                
            if DB_service.is_image_exist(db_connection,image_name):
                if not DB_service.delete_all_annote_by_imagename(db_connection,image_name):
                    return jsonify({'success': False,'error': f'Error when delete exist annote.'}), 402
                if not DB_service.create_annotes_by_iid(db_connection,DB_service.get_id_by_imagename(db_connection,image_name), label_list):
                    return jsonify({'success': False,'error': f'Error when create annote to exist image.'}), 402
                return jsonify({'success': True, 'message': f'Annotated image has editted.'}), 201
            
            if not minio_client.stat_object('unannotated', image_name):
                return jsonify({'success': False,'error': 'Not found image.'}), 402
            
            response = minio_client.get_object('unannotated', image_name)
            data = BytesIO(response.read())
            nparr = np.frombuffer(data.getvalue(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image_height, image_width, _ = image.shape

            minio_client.copy_object('annotated',image_name, CopySource('unannotated',image_name))
            minio_client.remove_object('unannotated',image_name)

            if not DB_service.create_annotated_image(conn=db_connection,image_name=image_name,width=image_width, height=image_height, label_list=label_list):
                return jsonify({'success': False,'error': 'Create annotated image failed.'}), 402
            
            return jsonify({'success': True, 'message': f'Annotated image create successfully.'}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'success': False,'error': 'Invalid request method'}), 405

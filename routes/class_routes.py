from flask import Blueprint, request, jsonify, g
from minio import S3Error
from util import Util
from db_service import DB_service
class_routes = Blueprint('class_routes', __name__,)

@class_routes.route('/api/class', methods=['GET','POST', 'DELETE'])
def manage_class():
    db_connection = g.db_connection
    minio_client = g.minio_client
    if request.method == 'GET':     
        # List all buckets and object counts
        try:
            classes = DB_service.get_classes_name(db_connection)
            return jsonify({
                "classes" : classes,
            }), 200
        except S3Error as e:
            return jsonify({'success': False, 'error': str(e)}), 500
       
    elif request.method == 'POST':
        data = request.get_json()
        class_name = data.get('bucket_name')
        
        if not class_name:
            return jsonify({'success': False,'error': 'Class name is required'}), 400
        if not Util.is_valid_bucket_name(class_name):
            return jsonify({'success': False,'error': 'Invalid class name'}), 402
        try:
            if class_name in DB_service.get_classes_name(db_connection):
                return jsonify({'success': False, 'error': f'Already have {class_name}.'}), 402
            if(not DB_service.create_class(db_connection,class_name)):
                return jsonify({'success': False, 'error': f'Can not create class {class_name}.'}), 402
            return jsonify({'success': True, 'message': f'Class {class_name} created successfully.'}), 201
        except S3Error as e:
            return jsonify({'success': False,'error': str(e)}), 500

    elif request.method == 'DELETE':
        data = request.get_json()
        class_name = data.get('bucket_name')
        if not class_name:
            return jsonify({'success': False,'error': 'Bucket name is required'}), 400

        try:
            if(not DB_service.delete_class(db_connection,class_name,minio_client)):
                return jsonify({'success': False, 'error': f'Can not delete class {class_name}.'}), 402

            return jsonify({'success': True, 'message': f'Bucket {class_name} and all its contents deleted successfully.'}), 200
        except S3Error as e:
            return jsonify({'success': False,'error': str(e)}), 500

    else:
        return jsonify({'success': False,'error': 'Invalid request method'}), 405

import os
from flask import Blueprint, make_response, request, jsonify, g
from minio import S3Error
from util import Util
from db_service import DB_service

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
images_routes = Blueprint('images_routes', __name__)

@images_routes.route('/api/images', methods=['GET','POST', 'DELETE'])
async def handle_images():
    minio_client = g.minio_client
    bucket_name = "unannotated"
    if request.method == 'POST':
        # Check if the bucket exists
        try:
            found = minio_client.bucket_exists(bucket_name)
            if not found:
                return jsonify({'success': False, 'message': f'Bucket {bucket_name} not found'}), 401
        except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500      
         
        # Handle file uploads
        files = request.files.getlist("files[]")

        if not files:
            return jsonify({'success': False, 'errors': "no selected files"}), 402

        uploaded_files = []

        for file in files:
            if file.filename == "":
                return jsonify({'success': False, 'errors': "one or more files has no name"}), 403

            if Util.allowed_file(file.filename, ALLOWED_EXTENSIONS):
                image_name = file.filename
                size = os.fstat(file.fileno()).st_size

                # Check if the image name already exists in the bucket
                objects = minio_client.list_objects(bucket_name)
                objects_name = [obj.object_name for obj in objects]
                
                while any(image_name == obj_name for obj_name in objects_name):
                    image_name = Util.rename_dupplicate_image_name(image_name)
                Util.upload_object(minio_client, image_name, file, size, bucket_name)
                uploaded_files.append(image_name)
            else :
                return jsonify({'success': False, 'errors': "Wrong format."}), 404

            
        if uploaded_files:
            return jsonify({'success': True, 'uploaded_files': uploaded_files, "minio": f"Files {', '.join(uploaded_files)} are successfully uploaded to bucket {bucket_name}."}), 200
        else:
            return jsonify({'success': False, 'errors': "no files were uploaded"}), 404

    elif request.method == 'GET':
        # Generate presigned URLs for objects in the bucket
        object_data = Util.list_images(minio_client, bucket_name)
        return jsonify(object_data)

    elif request.method == 'DELETE':
        # Delete objects from the bucket
        try:
            # Check if the bucket exists
            try:
                found = minio_client.bucket_exists(bucket_name)
                if not found:
                    return jsonify({'success': False, 'message': f'Bucket {bucket_name} not found'}), 403
            except Exception as e:
                 return jsonify({'success': False, 'error': str(e)}), 500
            
            # Get the list of object names from the request body
            image_names = request.json.get('image_names', [])

            if not image_names:
                return jsonify({'success': False, 'message': 'No object names provided'}), 400

            # Check if all object names exist in the bucket
            missing_images = []
            for image_name in image_names:
                try:
                    minio_client.stat_object(bucket_name, image_name)
                except Exception as e:
                    missing_images.append(image_name)

            if missing_images:
                return jsonify({
                    'success': False,
                    'message': f'Objects {", ".join(missing_images)} not found in bucket {bucket_name}'
                }), 404

            deleted_images = []
            for image_name in image_names:
                # Remove the object from the bucket
                minio_client.remove_object(bucket_name, image_name)
                deleted_images.append(image_name)

            # Return a success response
            return jsonify({
                'success': True,
                'message': f'Objects {", ".join(deleted_images)} deleted from bucket {bucket_name}'
            }), 200

        except Exception as e:
            # Handle any exceptions that occurred
            return jsonify({'success': False, 'error': str(e)}), 500

    else:
        return make_response(jsonify({'success': False, 'errors': "invalid request method"}), 405)
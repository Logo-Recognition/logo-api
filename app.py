import os
import datetime
from typing import Optional
from flask import Flask, jsonify, g
from flask_cors import CORS
from minio import Minio,S3Error
import psycopg2
from psycopg2.extensions import connection
from ultralytics import RTDETR, YOLO
from flask_apscheduler import APScheduler
import init_db
from routes.scrape_routes import scrape_routes
from routes.class_routes import class_routes
from routes.images_routes import images_routes
from routes.dataset_routes import dataset_routes
from routes.annotated_images_routes import annotated_images_routes
from routes.model_routes import model_routes
import json
# ----------------------------
# Configuration
# ----------------------------
MINIO_API_HOST = f"{os.getenv('MINIO_HOST')}:{os.getenv('MINIO_API_PORT')}"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
INIT_BUCKET_NAME = ["unannotated","annotated"]

DB_CONFIG = {
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
}
TEMP_DIR = "temp_images"
TIMEOUT_SECONDS = 60 * 60  # 1 hour

# ----------------------------
# Create Flask App
# ----------------------------
app = Flask(__name__)
CORS(app)
scheduler = APScheduler()
app.config["WTF_CSRF_ENABLED"] = False
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

# ----------------------------
# Load Resources in App Context
# ----------------------------
with app.app_context():
    #Connect DB
    try:
        app.db_connection = psycopg2.connect(**DB_CONFIG)
        print("Connected to PostgreSQL database")
        init_db.init_db(app.db_connection)
    except (Exception, psycopg2.Error) as error:
        print("Error connecting to PostgreSQL database:", error)
        app.db_connection = None
    
    app.minio_client = Minio(MINIO_API_HOST, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, secure=False)

       # Initialize MinIO buckets
    
    for bucket_name in INIT_BUCKET_NAME:
        try:
            # Check if the bucket exists
            if not app.minio_client.bucket_exists(bucket_name):
                print(f"Bucket '{bucket_name}' does not exist. Creating it now...")
                app.minio_client.make_bucket(bucket_name)
                print(f"Bucket '{bucket_name}' created successfully.")
            else:
                print(f"Bucket '{bucket_name}' already exists.")

            # Define public access policy
            public_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                    }
                ]
            }

            # Apply the public policy
            app.minio_client.set_bucket_policy(bucket_name, json.dumps(public_policy))
            print(f"Bucket '{bucket_name}' is now public.")

        except S3Error as e:
            print(f"Error with bucket '{bucket_name}': {e}")


# ----------------------------
# Utility Functions
# ----------------------------
def delete_old_temp_files():
    """Delete temporary files older than TIMEOUT_SECONDS."""
    if not os.path.exists(TEMP_DIR):
        return
    now = datetime.datetime.now()
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        file_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
        if (now - file_creation_time).total_seconds() > TIMEOUT_SECONDS:
            print("Deleting old file:", file_path)
            os.remove(file_path)

def get_minio_client() -> Optional[Minio]:
    """Retrieve MinIO client from request context (g)."""
    if "minio_client" not in g:
        g.minio_client = app.minio_client
    return g.minio_client

def get_db_connection() -> Optional[connection]:
    """Retrieve database connection from request context (g)."""
    if "db_connection" not in g:
        g.db_connection = psycopg2.connect(**DB_CONFIG)
    return g.db_connection

# ----------------------------
# Flask Hooks
# ----------------------------
@app.before_request
def before_request():
    """Attach resources to request context."""
    get_minio_client()
    get_db_connection()

@app.teardown_appcontext
def close_db_connection(exception=None):
    """Close database connection when app shuts down."""
    db_conn = getattr(g, "db_connection", None)
    if db_conn is not None:
        db_conn.close()

# ----------------------------
# Background Task (Scheduler)
# ----------------------------
@scheduler.task("interval", id="delete_old_temp_files_task", seconds=300)
def scheduled_task():
    delete_old_temp_files()

scheduler.init_app(app)
scheduler.start()

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def health():
    """Health check endpoint."""
    if app.db_connection is None:
        return jsonify({"error": "Database connection is not available"}), 500
    return jsonify({"success": True})

app.register_blueprint(class_routes)
app.register_blueprint(images_routes)
app.register_blueprint(dataset_routes)
app.register_blueprint(annotated_images_routes)
app.register_blueprint(model_routes)
app.register_blueprint(scrape_routes)

# ----------------------------
# Run Application
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)

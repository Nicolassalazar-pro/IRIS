from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from contextlib import contextmanager
from sklearn.cluster import DBSCAN
from pymongo import MongoClient
import face_recognition
from PIL import Image
import numpy as np
import datetime
import hashlib
import logging
import string
import random
import msvcrt
import json
import time
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clustering.log'),
        logging.StreamHandler()
    ]
)

PROFILE_DIR = "../Profile_Images"
CACHE_DIR = "../Cache"
DATABASE_NAME = "context"
COLLECTION_NAME = "profiles"
CONFIG_FILE = "../config.txt"

IMAGE_CONFIG = {
    'FACE_SIMILARITY_THRESHOLD': 0.5,  # For face matching
    'VALID_FORMATS': {'.png', '.jpg', '.jpeg', '.gif', '.bmp'},
    'JITTER_AMOUNT': 10,  # For face encoding
    'MIN_CLUSTER_SIZE': 1,
    'MAX_CLUSTER_DISTANCE': 0.8,
}

@contextmanager
def file_lock(file_obj):
    """Windows-specific context manager for file locking"""
    try:
        while True:
            try:
                # Try to acquire lock
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                # Wait and retry if file is locked
                time.sleep(0.1)
        yield
    finally:
        try:
            # Release lock
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

def safe_read_json(file_path, default_value=None, max_retries=3, retry_delay=0.1):
    """
    Safely read JSON from a file with Windows-specific locking
    """
    if not os.path.exists(file_path):
        return default_value

    for attempt in range(max_retries):
        try:
            with open(file_path, 'r') as f:
                with file_lock(f):
                    return json.load(f)
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to read {file_path} after {max_retries} attempts: {e}")
                return default_value
            time.sleep(retry_delay)

def safe_write_json(file_path, data, max_retries=3, retry_delay=0.1):
    """
    Safely write JSON to a file with Windows-specific locking
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    temp_file = f"{file_path}.tmp"
    
    for attempt in range(max_retries):
        try:
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                with file_lock(f):
                    json.dump(data, f, indent=4)
            
            # Atomic rename to target file
            if os.path.exists(file_path):
                os.remove(file_path)  # Windows requires removing the destination file first
            os.rename(temp_file, file_path)
            return True
            
        except Exception as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
            if attempt == max_retries - 1:
                logging.error(f"Failed to write {file_path} after {max_retries} attempts: {e}")
                return False
                
            time.sleep(retry_delay)
    
    return False

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as file:
            config = json.loads(file.read())
            if 'MONGODB_URI' not in config:
                raise ValueError("MONGODB_URI not found in config.txt")
            return config['MONGODB_URI']
    except FileNotFoundError:
        default_config = {'MONGODB_URI': ''}
        with open(CONFIG_FILE, 'w') as file:
            json.dump(default_config, file, indent=4)
        logging.error(f"Created {CONFIG_FILE} - please add your MongoDB URI to it.")
        exit(1)
    except Exception as e:
        logging.error(f"Error with config file: {e}")
        exit(1)

def connect_to_mongodb(uri):
    try:
        client = MongoClient(uri)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        client.admin.command('ping')
        logging.info("Connected to MongoDB Atlas successfully!")
        return collection
    except Exception as e:
        logging.error(f"Error connecting to MongoDB Atlas: {e}")
        exit(1)

def generate_random_hash(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def validate_image(image_path):
    """
    Validate that image is 216x216 and contains a face.
    Deletes invalid images.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (bool, str) - (is_valid, error_message)
    """
    try:
        # Check file exists
        if not os.path.exists(image_path):
            return False, "File does not exist"
            
        # Check dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            if width != 216 or height != 216:
                error_msg = f"Image must be 216x216, got {width}x{height}"
                logging.warning(f"Deleting invalid image {image_path}: {error_msg}")
                os.remove(image_path)
                return False, error_msg
        
        # Check for face
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            error_msg = "No face detected in image"
            logging.warning(f"Deleting invalid image {image_path}: {error_msg}")
            os.remove(image_path)
            return False, error_msg
            
        return True, ""
        
    except Exception as e:
        error_msg = f"Image validation failed: {str(e)}"
        if os.path.exists(image_path):
            logging.warning(f"Deleting invalid image {image_path}: {error_msg}")
            os.remove(image_path)
        return False, error_msg

def validate_cluster_quality(encodings, labels, cluster_label):
    """
    Validate the quality of a cluster by checking average distances between points.
    
    Args:
        encodings: List of face encodings
        labels: List of cluster labels from DBSCAN
        cluster_label: The specific cluster label to validate
    
    Returns:
        bool: True if cluster quality is acceptable, False otherwise
    """
    if cluster_label == -1:  # Noise points
        return False
        
    cluster_encodings = [enc for i, enc in enumerate(encodings) if labels[i] == cluster_label]
    if len(cluster_encodings) < IMAGE_CONFIG['MIN_CLUSTER_SIZE']:
        return False
        
    # Calculate average distance between all points in cluster
    distances = []
    for i in range(len(cluster_encodings)):
        for j in range(i + 1, len(cluster_encodings)):
            distance = np.linalg.norm(cluster_encodings[i] - cluster_encodings[j])
            distances.append(distance)
            
    if not distances:  # Single-point cluster
        return True
        
    avg_distance = sum(distances) / len(distances)
    return avg_distance <= IMAGE_CONFIG['MAX_CLUSTER_DISTANCE']

def get_face_encoding(image_path):
    """
    Get face encoding from image.
    """
    try:
        # Validate image first
        is_valid, error_msg = validate_image(image_path)
        if not is_valid:
            logging.error(f"Invalid image {image_path}: {error_msg}")
            if os.path.exists(image_path):
                os.remove(image_path)
            return None

        # Get face encoding
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if face_encodings:
            return face_encodings[0]
        
        logging.warning(f"Face found but couldn't encode: {image_path}")
        os.remove(image_path)
        return None
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        if os.path.exists(image_path):
            os.remove(image_path)
        return None

def process_images_batch(image_paths, existing_groups, processed_hashes, collection, watch_path):
    """
    Process a batch of images, first trying to match with existing groups from JSON
    before creating new groups.
    """
    logging.info(f"Processing batch of {len(image_paths)} images")
    new_encodings = {}
    cache_path = os.path.join(CACHE_DIR, "face_groups.json")
    
    # Safely read the existing face groups
    face_groups = safe_read_json(cache_path, default_value={})
    
    # Load existing group encodings from files in face_groups.json
    for group_id, filenames in face_groups.items():
        if group_id not in existing_groups:
            existing_groups[group_id] = []
            for filename in filenames:
                # Get full path for the image
                img_path = os.path.join(watch_path, group_id, filename)
                if os.path.exists(img_path):
                    encoding = get_face_encoding(img_path)
                    if encoding is not None:
                        existing_groups[group_id].append(encoding)
   
    # Filter out non-existent images
    valid_image_paths = [path for path in image_paths if os.path.exists(path)]
    if len(valid_image_paths) != len(image_paths):
        logging.warning(f"Skipped {len(image_paths) - len(valid_image_paths)} non-existent images")
   
    # Process each new image
    for image_path in valid_image_paths:
        try:
            # Check if image was already processed
            img_hash = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
            if img_hash in processed_hashes:
                continue
            processed_hashes.add(img_hash)
            
            # Validate image dimensions and face presence
            is_valid, error_msg = validate_image(image_path)
            if not is_valid:
                logging.warning(f"Skipping invalid image {image_path}: {error_msg}")
                continue
               
            # Get face encoding
            encoding = get_face_encoding(image_path)
            if encoding is not None:
                new_encodings[image_path] = encoding
               
        except Exception as e:
            logging.error(f"Error processing {image_path}: {e}")
            continue
   
    if not new_encodings:
        return existing_groups

    # Try matching with ALL existing groups first
    unmatched_encodings = {}
    unmatched_paths = []
    
    for new_path, new_encoding in new_encodings.items():
        filename = os.path.basename(new_path)
        matched = False
        
        # Check against ALL existing groups from face_groups.json
        for group_id, group_encodings in existing_groups.items():
            if not group_encodings:  # Skip empty groups
                continue
                
            for group_encoding in group_encodings:
                distance = np.linalg.norm(new_encoding - group_encoding)
                if distance < IMAGE_CONFIG['FACE_SIMILARITY_THRESHOLD']:
                    # Add to existing group
                    if group_id not in face_groups:
                        face_groups[group_id] = []
                    if filename not in face_groups[group_id]:
                        face_groups[group_id].append(filename)
                        existing_groups[group_id].append(new_encoding)
                    matched = True
                    logging.info(f"Matched {filename} to existing group {group_id}")
                    break
            if matched:
                break
        
        # If no match found in any existing group, add to unmatched
        if not matched:
            unmatched_encodings[new_path] = new_encoding
            unmatched_paths.append(new_path)
            logging.info(f"No match found for {filename} in existing groups")

    # Only run DBSCAN on truly unmatched images
    if unmatched_encodings:
        try:
            encodings_list = list(unmatched_encodings.values())
            clustering = DBSCAN(
                eps=IMAGE_CONFIG['FACE_SIMILARITY_THRESHOLD'],
                min_samples=IMAGE_CONFIG['MIN_CLUSTER_SIZE'],
                metric='euclidean'
            ).fit(encodings_list)
            
            # Group images by cluster label
            clusters = {}
            for idx, label in enumerate(clustering.labels_):
                if validate_cluster_quality(encodings_list, clustering.labels_, label):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(unmatched_paths[idx])
            
            # Create new groups only for images that couldn't match existing ones
            for label, paths in clusters.items():
                if label != -1:  # Skip noise points
                    new_group_id = f"PersonID_{generate_random_hash()}"
                    face_groups[new_group_id] = []
                    existing_groups[new_group_id] = []
                    
                    for path in paths:
                        filename = os.path.basename(path)
                        face_groups[new_group_id].append(filename)
                        existing_groups[new_group_id].append(unmatched_encodings[path])
                        logging.info(f"Added {filename} to new cluster group {new_group_id}")
                    
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
    
    # Clean up and write to cache
    if safe_write_json(cache_path, face_groups):
        logging.info("Updated face groups cache")
    else:
        logging.error("Failed to update face groups cache")
   
    # Create profiles for new groups
    create_profiles_from_json(collection)
   
    return existing_groups

def process_initial_images(watch_path, collection):
    logging.info(f"Processing initial images in {watch_path}")
    processed_hashes = set()
    existing_groups = {}
    cache_path = os.path.join(CACHE_DIR, "face_groups.json")

    # First load existing groups from JSON
    face_groups = safe_read_json(cache_path, default_value={})
    if face_groups:
        logging.info(f"Found existing face groups in JSON: {len(face_groups)} groups")
        # Load encodings for existing groups from JSON
        for group_id, filenames in face_groups.items():
            existing_groups[group_id] = []
            group_dir = os.path.join(watch_path, group_id)
            if os.path.exists(group_dir):
                for filename in filenames:
                    img_path = os.path.join(group_dir, filename)
                    if os.path.exists(img_path):
                        encoding = get_face_encoding(img_path)
                        if encoding is not None:
                            existing_groups[group_id].append(encoding)
                            # Add to processed hashes to avoid reprocessing
                            with open(img_path, 'rb') as f:
                                img_hash = hashlib.md5(f.read()).hexdigest()
                                processed_hashes.add(img_hash)
    
    # Then process any new images in the root directory
    image_paths = []
    for filename in os.listdir(watch_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            if not any(filename in group_files for group_files in face_groups.values()):
                image_paths.append(os.path.join(watch_path, filename))
    
    if image_paths:
        logging.info(f"Found {len(image_paths)} new images to process")
        existing_groups = process_images_batch(image_paths, existing_groups, processed_hashes, collection, watch_path)
    else:
        logging.info("No new images found to process")
    
    return existing_groups, processed_hashes

def create_profiles_from_json(collection):
    """Create MongoDB profiles for PersonIDs in face_groups.json."""
    created_count = 0
    errors = []
    cache_path = os.path.join(CACHE_DIR, "face_groups.json")
    
    # Safely read the face groups
    face_groups = safe_read_json(cache_path, default_value=None)
    if face_groups is None:
        return 0, ["face_groups.json not found or invalid"]
            
    for person_id in face_groups:
        try:
            if not collection.find_one({"_id": person_id}):
                new_profile = {
                    "_id": person_id,
                    "first_name": "",
                    "last_name": "",
                    "age": "",
                    "context": "",
                    "chat_history": [],
                    "created_at": datetime.datetime.utcnow().astimezone(datetime.timezone.utc).isoformat(),
                    "updated_at": datetime.datetime.utcnow().astimezone(datetime.timezone.utc).isoformat()
                }
                result = collection.insert_one(new_profile)
                
                if result.inserted_id:
                    created_count += 1
                    logging.info(f"Created new profile for '{person_id}'")
                else:
                    errors.append(f"Failed to create profile for '{person_id}'")
                    
        except Exception as e:
            error_msg = f"Error creating profile for '{person_id}': {str(e)}"
            logging.error(error_msg)
            errors.append(error_msg)
                
    return created_count, errors

def create_event_handler(watch_path, collection):
    existing_groups, processed_hashes = process_initial_images(watch_path, collection)
    processing = False
    pending_images = []
    last_process_time = time.time()
    
    class ImageEventHandler(FileSystemEventHandler):
        def on_created(self, event):
            nonlocal processing, existing_groups, pending_images, last_process_time
            
            if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                if "PersonID_" in event.src_path:
                    return
                
                time.sleep(0.2)
                
                if not os.path.exists(event.src_path):
                    logging.warning(f"File disappeared before processing: {event.src_path}")
                    return
                
                try:
                    with open(event.src_path, 'rb') as f:
                        f.read(1)
                except Exception as e:
                    logging.warning(f"File not ready for processing: {event.src_path} - {e}")
                    return
                    
                if event.src_path not in pending_images:
                    logging.info(f"New image detected: {event.src_path}")
                    pending_images.append(event.src_path)
                
                current_time = time.time()
                if not processing and (current_time - last_process_time >= 0.5):
                    processing = True
                    try:
                        if pending_images:
                            valid_pending = [p for p in pending_images if os.path.exists(p)]
                            if len(valid_pending) != len(pending_images):
                                logging.warning(f"Skipped {len(pending_images) - len(valid_pending)} missing files")
                            
                            root_path = os.path.dirname(pending_images[0])
                            all_images = [
                                os.path.join(root_path, f) 
                                for f in os.listdir(root_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
                                and "PersonID_" not in f
                                and os.path.exists(os.path.join(root_path, f))
                            ]
                            images_to_process = list(set(valid_pending + all_images))
                            pending_images.clear()
                            
                            if images_to_process:
                                logging.info(f"Processing batch of {len(images_to_process)} images")
                                existing_groups = process_images_batch(
                                    images_to_process,
                                    existing_groups,
                                    processed_hashes,
                                    collection,
                                    watch_path
                                )
                                last_process_time = current_time
                            
                    except Exception as e:
                        logging.error(f"Error processing images: {e}")
                    finally:
                        processing = False
    
    return ImageEventHandler()

def main():
    mongo_uri = load_config()
    if not mongo_uri:
        logging.error("Please add your MongoDB URI to config.txt")
        exit(1)
        
    logging.info("Connecting to MongoDB...")
    collection = connect_to_mongodb(mongo_uri)
    logging.info("Connected to MongoDB!")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    watch_path = os.path.join(current_dir, PROFILE_DIR)
    
    if not os.path.exists(watch_path):
        os.makedirs(watch_path)
        logging.info(f"Created profile directory: {watch_path}")
    
    # Create profiles from JSON before starting event handler
    event_handler = create_event_handler(watch_path, collection)
    profiles_created, errors = create_profiles_from_json(collection)
    
    if profiles_created > 0:
        logging.info(f"Successfully created {profiles_created} new profile(s)")
    if errors:
        logging.warning("Errors encountered during profile creation:")
        for error in errors:
            logging.warning(f"  - {error}")
    
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    observer.start()
    
    logging.info(f"Monitoring {watch_path} for new images...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping monitoring...")
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
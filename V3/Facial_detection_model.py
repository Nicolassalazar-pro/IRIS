from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from contextlib import contextmanager
from sklearn.cluster import DBSCAN
from pymongo import MongoClient
import sounddevice as sd
from pathlib import Path
import face_recognition
from PIL import Image
import numpy as np
import contextlib
import threading
import keyboard
import datetime
import hashlib
import logging
import pyttsx3
import whisper
import string
import random
import pickle
import ollama
import msvcrt
import queue
import torch
import json
import time
import wave
import pytz
import cv2
import sys
import os

CONFIG_FILE = "config.py"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('FD.log'),
        #logging.StreamHandler()
    ]
)

try:
    with open(CONFIG_FILE, 'r') as file:
        config = eval(file.read())
        if 'MONGODB_URI' not in config:
            raise ValueError("MONGODB_URI not found in config.txt")
except FileNotFoundError:
    logging.error(f"{CONFIG_FILE} NotFound - please download or Create one nerd")
    exit(1)
except Exception as e:
    logging.error(f"Error with config file: {e}")
    exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PERSONALITY = config["FRIDAY"]

# Create necessary directories
PROFILE_DIR = Path(config["PROFILE_DIR"])
CACHE_DIR = Path(config["CACHE_DIR"])

PROFILE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

DTYPE = np.int16
q = queue.Queue()
chat_history = []
Current_Person = 'PersonID_XKl6SYJx'

#os.system("cls")

#print("\n=== Initializing Friday Voice Chat System ===")
#print("📝 Loading Whisper model...")
whisper_model = whisper.load_model(config["WHISPER_SIZE"], device=DEVICE)

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
        db = client[config["DATABASE_NAME"]]
        collection = db[config["COLLECTION_NAME"]]
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
    if len(cluster_encodings) < config['MIN_CLUSTER_SIZE']:
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
    return avg_distance <= config['MAX_CLUSTER_DISTANCE']

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
    cache_path = os.path.join(CACHE_DIR, config["GROUP_JSON_NAME"])
    
    # Safely read the existing face groups
    face_groups = safe_read_json(cache_path, default_value={})
    
    # First, check for duplicate groups in existing data
    groups_to_merge = {}  # old_id -> new_id
    for group1_id, group1_encodings in existing_groups.items():
        if not group1_encodings or group1_id in groups_to_merge:
            continue
            
        # Compare with other groups
        for group2_id, group2_encodings in existing_groups.items():
            if (group2_id == group1_id or 
                not group2_encodings or 
                group2_id in groups_to_merge):
                continue
                
            # Use existing similarity check logic
            for enc1 in group1_encodings:
                match_found = False
                for enc2 in group2_encodings:
                    distance = np.linalg.norm(enc1 - enc2)
                    if distance < config['FACE_SIMILARITY_THRESHOLD']:
                        groups_to_merge[group2_id] = group1_id
                        logging.info(f"Found duplicate groups: merging {group2_id} into {group1_id}")
                        match_found = True
                        break
                if match_found:
                    break
    
    # If groups need to be merged, update face_groups and MongoDB
    if groups_to_merge:
        merged_face_groups = {}
        for group_id, filenames in face_groups.items():
            if group_id in groups_to_merge:
                # Add files to the target group
                target_group = groups_to_merge[group_id]
                if target_group not in merged_face_groups:
                    merged_face_groups[target_group] = []
                merged_face_groups[target_group].extend(filenames)
            else:
                merged_face_groups[group_id] = filenames
        
        # Remove duplicates from merged groups
        for group_id in merged_face_groups:
            merged_face_groups[group_id] = list(set(merged_face_groups[group_id]))
        
        # Update face_groups with merged version
        face_groups = merged_face_groups
        
        # Update MongoDB - merge profiles for merged groups
        for old_group, new_group in groups_to_merge.items():
            try:
                old_profile = collection.find_one({"_id": old_group})
                if old_profile:
                    new_profile = collection.find_one({"_id": new_group})
                    if new_profile:
                        # Merge profile data
                        updates = {}
                        for field in ["first_name", "last_name", "age", "context"]:
                            if old_profile.get(field) and not new_profile.get(field):
                                updates[field] = old_profile[field]
                        
                        if old_profile.get("chat_history"):
                            updates["chat_history"] = new_profile.get("chat_history", []) + old_profile["chat_history"]
                        
                        if updates:
                            collection.update_one(
                                {"_id": new_group},
                                {"$set": updates}
                            )
                    
                    # Delete old profile
                    collection.delete_one({"_id": old_group})
                    logging.info(f"Merged and deleted profile for '{old_group}'")
            except Exception as e:
                logging.error(f"Error merging profiles: {e}")
        
        # Save cleaned groups back to JSON
        if safe_write_json(cache_path, face_groups):
            logging.info("Updated face groups cache after merging duplicates")
        else:
            logging.error("Failed to update face groups cache after merging")
            
    # Load existing group encodings from files in face_groups.json
    existing_groups = {}  # Reset existing_groups to reflect merged groups
    for group_id, filenames in face_groups.items():
        existing_groups[group_id] = []
        for filename in filenames:
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
                if distance < config['FACE_SIMILARITY_THRESHOLD']:
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
                eps=config['FACE_SIMILARITY_THRESHOLD'],
                min_samples=config['MIN_CLUSTER_SIZE'],
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
    profile_results = create_profiles_from_json(collection)
    if profile_results['created'] > 0:
        logging.info(f"Created {profile_results['created']} new profile(s)")
    if profile_results['deleted'] > 0:
        logging.info(f"Removed {profile_results['deleted']} obsolete profile(s)")
    if profile_results['errors']:
        for error in profile_results['errors']:
            logging.warning(f"Profile error: {error}")
   
    return existing_groups

def process_initial_images(watch_path, collection):
    logging.info(f"Processing initial images in {watch_path}")
    processed_hashes = set()
    existing_groups = {}
    cache_path = os.path.join(CACHE_DIR, config["GROUP_JSON_NAME"])

    # First load existing groups from JSON
    face_groups = safe_read_json(cache_path, default_value={})
    if face_groups:
        logging.info(f"Found existing face groups in JSON: {len(face_groups)} groups")
        
        # Check for duplicate images across groups before loading encodings
        image_to_groups = {}  # Keep track of which images appear in which groups
        groups_to_merge = {}  # old_id -> new_id
        
        # Build image to groups mapping
        for group_id, filenames in face_groups.items():
            for filename in filenames:
                if filename not in image_to_groups:
                    image_to_groups[filename] = []
                image_to_groups[filename].append(group_id)
        
        # For any image that appears in multiple groups, merge those groups
        for filename, group_list in image_to_groups.items():
            if len(group_list) > 1:
                # Use the first group as the target group
                target_group = group_list[0]
                # Mark all other groups for merging into the target group
                for other_group in group_list[1:]:
                    if other_group not in groups_to_merge:
                        groups_to_merge[other_group] = target_group
                        logging.info(f"Startup: Found duplicate image {filename}: merging {other_group} into {target_group}")

        # If groups need to be merged, update face_groups and MongoDB
        if groups_to_merge:
            merged_face_groups = {}
            for group_id, filenames in face_groups.items():
                if group_id in groups_to_merge:
                    # Add files to the target group
                    target_group = groups_to_merge[group_id]
                    if target_group not in merged_face_groups:
                        merged_face_groups[target_group] = []
                    merged_face_groups[target_group].extend(filenames)
                else:
                    merged_face_groups[group_id] = filenames
            
            # Remove duplicates from merged groups
            for group_id in merged_face_groups:
                merged_face_groups[group_id] = list(set(merged_face_groups[group_id]))
            
            # Update face_groups with merged version
            face_groups = merged_face_groups
            
            # Save cleaned groups back to JSON - THIS WAS MISSING
            if not safe_write_json(cache_path, face_groups):
                logging.error("Startup: Failed to save cleaned groups to JSON")
                return existing_groups, processed_hashes
            
            # Update MongoDB - merge profiles for merged groups
            for old_group, new_group in groups_to_merge.items():
                try:
                    old_profile = collection.find_one({"_id": old_group})
                    if old_profile:
                        new_profile = collection.find_one({"_id": new_group})
                        if new_profile:
                            updates = {}
                            for field in ["first_name", "last_name", "age", "context"]:
                                if old_profile.get(field) and not new_profile.get(field):
                                    updates[field] = old_profile[field]
                            
                            if old_profile.get("chat_history"):
                                updates["chat_history"] = new_profile.get("chat_history", []) + old_profile["chat_history"]
                            
                            if updates:
                                collection.update_one(
                                    {"_id": new_group},
                                    {"$set": updates}
                                )
                        
                        # Delete old profile
                        collection.delete_one({"_id": old_group})
                        logging.info(f"Startup: Merged and deleted profile for '{old_group}'")
                except Exception as e:
                    logging.error(f"Startup: Error merging profiles: {e}")

        # Now load encodings for cleaned groups
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
    """Create MongoDB profiles for PersonIDs in face_groups.json and remove obsolete ones."""
    created_count = 0
    deleted_count = 0
    errors = []
    cache_path = os.path.join(CACHE_DIR, config["GROUP_JSON_NAME"])
    
    # Safely read the face groups
    face_groups = safe_read_json(cache_path, default_value=None)
    if face_groups is None:
        return {
            'created': 0,
            'deleted': 0,
            'errors': ["face_groups.json not found or invalid"]
        }
    # Get all existing profiles from MongoDB
    existing_profiles = set(profile['_id'] for profile in collection.find({}, {'_id': 1}))
    
    # Get all valid PersonIDs from JSON
    valid_person_ids = set(face_groups.keys())
    
    # Find obsolete profiles (in MongoDB but not in JSON)
    obsolete_profiles = existing_profiles - valid_person_ids
    if obsolete_profiles:
        try:
            result = collection.delete_many({'_id': {'$in': list(obsolete_profiles)}})
            deleted_count = result.deleted_count
            logging.info(f"Removed {deleted_count} obsolete profiles from MongoDB")
        except Exception as e:
            error_msg = f"Error removing obsolete profiles: {str(e)}"
            logging.error(error_msg)
            errors.append(error_msg)
    
    # Create new profiles for PersonIDs in JSON but not in MongoDB
    for person_id in valid_person_ids:
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
    
    return {
        'created': created_count,
        'deleted': deleted_count,
        'errors': errors
    }

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

def find_person_id(json_data, target_filename):
    """
    Find the PersonID associated with a given filename in the JSON data.
    
    Args:
        json_data (dict): The JSON data containing PersonID groups and filenames
        target_filename (str): The filename to search for
        
    Returns:
        str: The PersonID if found, None if not found
    """
    for person_id, filenames in json_data.items():
        if target_filename in filenames:
            return person_id
    return None

def generate_face_id(length=8):
    """Generate a random ID for new faces"""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def get_enrolled_images():
    """Get list of all enrolled face images"""
    image_files = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    for ext in valid_extensions:
        image_files.extend(PROFILE_DIR.rglob(f"*{ext}"))
    return image_files

def enroll_face(frame, face_location, collection):
    """
    Save a new face image as a 216x216 square image
    """
    try:
        top, right, bottom, left = face_location
        
        # Calculate face dimensions
        face_height = bottom - top
        face_width = right - left
        
        # Make the selection square by using the larger dimension
        square_size = max(face_width, face_height)
        
        # Calculate center of the face
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        # Calculate new square boundaries with additional padding (20% of square_size)
        padding = int(square_size * 0.2)
        square_size_with_padding = square_size + (2 * padding)
        
        # Calculate new boundaries while maintaining the center
        new_left = center_x - (square_size_with_padding // 2)
        new_right = center_x + (square_size_with_padding // 2)
        new_top = center_y - (square_size_with_padding // 2)
        new_bottom = center_y + (square_size_with_padding // 2)
        
        # Ensure boundaries are within frame
        frame_height, frame_width = frame.shape[:2]
        new_left = max(0, new_left)
        new_right = min(frame_width, new_right)
        new_top = max(0, new_top)
        new_bottom = min(frame_height, new_bottom)
        
        # Extract face region
        face_image = frame[new_top:new_bottom, new_left:new_right]
        
        # If the crop resulted in a non-square image (due to frame boundaries),
        # create a square black background
        if face_image.shape[0] != face_image.shape[1]:
            size = max(face_image.shape[0], face_image.shape[1])
            square_image = np.zeros((size, size, 3), dtype=np.uint8)
            
            # Calculate position to paste the face image
            y_offset = (size - face_image.shape[0]) // 2
            x_offset = (size - face_image.shape[1]) // 2
            
            # Paste the face image onto the square background
            square_image[
                y_offset:y_offset + face_image.shape[0],
                x_offset:x_offset + face_image.shape[1]
            ] = face_image
            
            face_image = square_image
        
        # Resize to final 216x216 size
        face_image = cv2.resize(face_image, (216, 216), interpolation=cv2.INTER_LANCZOS4)
        
        # Save the image
        filename = f"ID_{generate_face_id()}.jpg"
        filepath = PROFILE_DIR / filename
        cv2.imwrite(str(filepath), face_image)
        logging.info(f"Enrolled new face: {filepath}")
        
        # Trigger clustering process for the new image
        existing_groups = {}
        processed_hashes = set()
        cache_path = os.path.join(CACHE_DIR, config["GROUP_JSON_NAME"])
        
        # Load existing groups
        face_groups = safe_read_json(cache_path, default_value={})
        if face_groups:
            for group_id, filenames in face_groups.items():
                existing_groups[group_id] = []
                for fname in filenames:
                    img_path = os.path.join(PROFILE_DIR, fname)
                    if os.path.exists(img_path):
                        encoding = get_face_encoding(img_path)
                        if encoding is not None:
                            existing_groups[group_id].append(encoding)
        
        # Process the new image
        process_images_batch(
            [str(filepath)],
            existing_groups,
            processed_hashes,
            collection,
            str(PROFILE_DIR)
        )
        
        return filepath
        
    except Exception as e:
        logging.error(f"Face enrollment failed: {str(e)}")
        return None

def generate_face_encodings():
    """Generate encodings for all enrolled faces"""
    face_encodings = []
    face_identifiers = []
    
    image_files = get_enrolled_images()
    if not image_files:
        logging.warning("No faces found to encode")
        return [], []

    for img_path in image_files:
        try:
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                face_encodings.append(encodings[0])
                face_identifiers.append(str(img_path.relative_to(PROFILE_DIR)))
                logging.info(f"Generated encoding for: {img_path.name}")
                
        except Exception as e:
            logging.error(f"Failed to encode {img_path}: {str(e)}")
            continue

    if face_encodings:
        # Cache the encodings
        cache_file = CACHE_DIR / config["ENCODING_NAME"]
        with open(cache_file, 'wb') as f:
            pickle.dump([face_encodings, face_identifiers], f)
        logging.info(f"Successfully encoded {len(face_encodings)} faces")
        
    return face_encodings, face_identifiers

def load_cached_encodings():
    """Load previously cached face encodings"""
    try:
        cache_file = CACHE_DIR / config["ENCODING_NAME"]
        with open(cache_file, 'rb') as f:
            face_encodings, face_identifiers = pickle.load(f)
        logging.info(f"Loaded {len(face_encodings)} cached encodings")
        return face_encodings, face_identifiers
    except Exception as e:
        logging.error(f"Failed to load cached encodings: {str(e)}")
        return [], []

def process_frame(frame, face_encodings, face_identifiers, awaiting_first_enrollment, collection):
    """Process a video frame for face detection and recognition"""
    
    if config['SHOW_CAMERA']:
        display_frame = frame.copy()
    else:
        display_frame = None
    
    first_face_enrolled = False
    
    # Scale down frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), None, config["PROCESSING_SCALE"], config["PROCESSING_SCALE"])
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)

    current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    cache_path = os.path.join(CACHE_DIR, config["GROUP_JSON_NAME"])
    
    
    for face_encoding, face_location in zip(current_face_encodings, face_locations):
        # Scale coordinates back to original size
        top, right, bottom, left = [int(coord / config["PROCESSING_SCALE"]) for coord in face_location]
        scaled_location = (top, right, bottom, left)
        
        if awaiting_first_enrollment:
            name = "First Face - Enrolling..."
            color = (255, 165, 0)  # Orange
            
            if enroll_face(frame, scaled_location, collection):
                face_encodings, face_identifiers = generate_face_encodings()
                first_face_enrolled = True
        else:
            # Check if we have any encodings to compare against
            if face_encodings:
                matches = face_recognition.compare_faces(face_encodings, face_encoding)
                if True in matches:
                    first_match_index = matches.index(True)
                    ID_groups = safe_read_json(cache_path, default_value={})
                    name = find_person_id(ID_groups, str(Path(face_identifiers[first_match_index])))
                    color = (0, 255, 0)  # Green
                else:
                    name = "Unknown - Enrolling..."
                    color = (0, 0, 255)  # Red
                    if enroll_face(frame, scaled_location, collection):
                        face_encodings, face_identifiers = generate_face_encodings()
            else:
                name = "Unknown - Enrolling..."
                color = (0, 0, 255)  # Red
                if enroll_face(frame, scaled_location, collection):
                    face_encodings, face_identifiers = generate_face_encodings()
        
        # Draw box and label based on configuration
        if config['SHOW_BOUNDING_BOX']:
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
        
        if config['SHOW_LABELS']:
            cv2.putText(display_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        Current_Person = name
    
    return display_frame, first_face_enrolled, face_encodings, face_identifiers

def face_detect(camera, face_encodings, face_identifiers, awaiting_first_enrollment, collection):
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        display_frame, first_face_enrolled, face_encodings, face_identifiers = process_frame(
            frame, face_encodings, face_identifiers, awaiting_first_enrollment, collection
        )
        
        if first_face_enrolled:
            awaiting_first_enrollment = False
            logging.info("First face enrolled! Continuing with recognition...")
        
        if config['SHOW_CAMERA'] and display_frame is not None:
            cv2.imshow("Face Recognition System", display_frame)
            cv2.waitKey(1)


def initialize_audio():
    """Initialize audio device with proper error handling"""
    try:
        # List available devices
        devices = sd.query_devices()
        # Find the default input device
        default_device = sd.default.device[0]  # Get default input device
        if default_device is None:
            # If no default device, try to find the first input device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    default_device = i
                    break
        if default_device is None:
            raise RuntimeError("No input device found")
        return default_device
    except Exception as e:
        logging.error(f"Error initializing audio: {e}")
        sys.exit(1)

def timestamp():
    """Get current Eastern time timestamp"""
    tz = pytz.timezone('US/Eastern')
    return datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S EST')

def get_ollama_response(messages,collection):
    try:

        Current_Person_Context = collection.find_one({"_id": Current_Person})

        context_messages = [
            {"role": "system", "content": f"This is your context window in regards to who you are speaking to. First name: {Current_Person_Context["first_name"]}; Last name: {Current_Person_Context["last_name"]}; Age : {Current_Person_Context["age"]}."},
            {"role": "system", "content": "If any of the previous fields are not filled, you should prompt the person for the information, But you can not prompt the user for more than one piece of information per question asked."},
            {"role": "system", "content": f"This is your context window in regards about the person you are speaking to. Context:{Current_Person_Context["context"]}"},
            {"role": "system", "content": PERSONALITY},
            {"role": "system", "content": "Remember: Keep responses short and direct. No emojis or multiple questions."},
            *messages
        ]
        
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                response = ollama.chat(
                    model = config["LLM_MODEL"],  # Changed model name
                    messages=context_messages,
                    options={
                        "mirostat": config["MIROSTAT"],
                        "mirostat_tau": config["MIROSTAT_TAU"],
                        "num_ctx": config["NUM_CTX"],
                        "num_thread": config["NUM_THREAD"],
                        "temperature": config["TEMPERATURE"],
                        "top_k": config["TOP_K"],
                        "top_p": config["TOP_P"]
                    }
                )
        return response['message']['content']
    except Exception as e:
        return f"Looks like we hit a snag: {str(e)}"

def audio_callback(indata):
    """Callback for audio recording"""
    if keyboard.is_pressed('space'):
        q.put(bytes(indata))

def save_audio_chunk(audio_data):
    """Save audio data to a temporary WAV file"""
    temp_filename = "temp_audio.wav"
    with wave.open(temp_filename, 'wb') as wf:
        wf.setnchannels(config["NCHANNELS"])
        wf.setsampwidth(config["SAMPWIDTH"])
        wf.setframerate(config["SAMPLE_RATE"])
        wf.writeframes(b''.join(audio_data))
    return temp_filename

def transcribe_audio(filename):
    """Transcribe audio using Whisper"""
    # Load audio using whisper's built-in audio loading
    result = whisper_model.transcribe(
        filename,
        fp16=True,  # Set to True if using GPU and want faster processing
        language='en',  # You can specify language if know
    )
    return result["text"]

def process_audio():
    """Process audio from the queue"""
    audio_chunks = []
    while True:
        try:
            data = q.get_nowait()
            audio_chunks.append(data)
        except queue.Empty:
            break
    
    if audio_chunks:
        temp_filename = save_audio_chunk(audio_chunks)
        text = transcribe_audio(temp_filename)
        os.remove(temp_filename)
        return text.strip()
    return ""

def TTS(text):
    """Text-to-speech conversion"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', .9)
    engine.say(text)
    engine.runAndWait()

def LLM(stream,collection):

    with stream:
        while True:
            if keyboard.is_pressed('space'):
                recording_start_time = time.time()
                while keyboard.is_pressed('space'):
                    elapsed_time = time.time() - recording_start_time
                    sys.stdout.write(f'\r Recording... [{elapsed_time:.1f}s] [SPACE held] ')
                    sys.stdout.flush()
                    time.sleep(0.1)
                
                sys.stdout.write('\r' + ' ' * 50 + '\r')
                sys.stdout.flush()
                
                print("\n Processing speech...")
                text = process_audio()
                
                if text:
                    print(f"\n You: {text}")
                    
                    chat_history.append({
                        "role": "user",
                        "content": text,
                        "timestamp": timestamp()
                    })

                    #Current_Person
                    messages = [
                        {"role": m["role"], "content": m["content"]} 
                        for m in chat_history
                    ]

                    logging.info(f"Messages:{messages}")
                    
                    print("Friday's thinking...")
                    response_start_time = time.time()
                    
                    assistant_response = get_ollama_response(messages,collection)
                    
                    response_time = time.time() - response_start_time
                    
                    print(f"\nFriday: {assistant_response}")
                    print(f"Response time: {response_time:.2f}s")

                    TTS(assistant_response)

                    chat_history.append({
                        "role": "assistant",
                        "content": assistant_response,
                        "timestamp": timestamp()
                    })
                    
                    with open('chat_history.json', 'w', encoding='utf-8') as f:
                        json.dump(chat_history, f, indent=2, ensure_ascii=False)
                    
                    print("\nReady for next input! (Hold SPACE to speak)")
            
            if keyboard.is_pressed('esc'):
                print("\nFriday signing off...")
                break
            
            time.sleep(0.01)

def main():
    mongo_uri = config['MONGODB_URI']
    if not mongo_uri:
        logging.error("Please add your MongoDB URI to config.py")
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
    profile_results = create_profiles_from_json(collection)
    
    if profile_results['created'] > 0:
        logging.info(f"Successfully created {profile_results['created']} new profile(s)")
    if profile_results['deleted'] > 0:
        logging.info(f"Removed {profile_results['deleted']} obsolete profile(s)")
    if profile_results['errors']:
        logging.warning("Errors encountered during profile operations:")
        for error in profile_results['errors']:
            logging.warning(f"  - {error}")

    # Try loading cached encodings first
    face_encodings, face_identifiers = load_cached_encodings()
    
    # If no cache, check enrollment directory
    if not face_encodings:
        face_encodings, face_identifiers = generate_face_encodings()

    awaiting_first_enrollment = not face_encodings
    if awaiting_first_enrollment:
        logging.warning("No enrolled faces found. Waiting for first face to enroll...")
    
    # Initialize camera
    camera = cv2.VideoCapture(config["CAPTURE_CAM_INDEX"])
    
    if not camera.isOpened():
        logging.error("Failed to access camera")
        return
        
    camera.set(3, config["CAPTURE_WIDTH"])
    camera.set(4, config["CAPTURE_HEIGHT"])

    # Set up audio stream
    input_device = initialize_audio()
    
    # Set up audio stream with explicit device
    stream = sd.RawInputStream(
        device=input_device,  # Use the found input device
        samplerate=config["SAMPLE_RATE"],
        blocksize=config["BLOCKSIZE"],
        dtype=DTYPE,
        channels= config["NCHANNELS"],
        callback=audio_callback
    )

    print("\n=== Friday Voice Chat Ready ===")
    print("\nSystem Status:")
    print(f'🎤 Speech: Whisper {config["WHISPER_SIZE"]} ({DEVICE} mode)')
    print(f'🤖 Chat: {config["LLM_MODEL"]} (GPU auto-detection)')
    
    print("\nControls:")
    print("- Press and HOLD SPACE to record speech")
    print("- Release SPACE to process and get response")
    print("- Press ESC to exit")
    print("\nFriday's ready to chat!")

    # Load existing chat history
    with open('chat_history.json', 'r', encoding='utf-8') as f:
        chat_history.extend(json.load(f))



    Clustering_Module = Observer()
    Clustering_Module.schedule(event_handler, watch_path, recursive=False)
    
    FaceID_Module = threading.Thread(target=face_detect, args=(camera, face_encodings, face_identifiers, awaiting_first_enrollment, collection))
    LLM_Module = threading.Thread(target=LLM, args=(stream,collection))

    logging.info(f"Monitoring {watch_path} for new images...")

    Clustering_Module.start()
    FaceID_Module.start()
    LLM_Module.start()

    try:
        while True:
            time.sleep(1)     
    except keyboard.is_pressed('esc'):
        logging.info("Stopping everything")
        camera.release()
        cv2.destroyAllWindows()
        Clustering_Module.stop()
        Clustering_Module.join()
        FaceID_Module.join()
        LLM_Module.join()

if __name__ == "__main__":
    main()
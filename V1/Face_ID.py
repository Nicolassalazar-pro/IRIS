from contextlib import contextmanager
from pathlib import Path
import face_recognition
import numpy as np
import pickle
import random
import string
import msvcrt
import json
import time
import cv2
import os

# Configuration
ENROLLMENT_DIR = Path("../Profile_Images")
CACHE_DIR = Path("../Cache")
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
PROCESSING_SCALE = 0.25

# Create necessary directories
ENROLLMENT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

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
                print(f"Failed to read {file_path} after {max_retries} attempts: {e}")
                return default_value
            time.sleep(retry_delay)

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
        image_files.extend(ENROLLMENT_DIR.rglob(f"*{ext}"))
    return image_files

def enroll_face(frame, face_location):
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
        filepath = ENROLLMENT_DIR / filename
        cv2.imwrite(str(filepath), face_image)
        print(f"Enrolled new face: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Face enrollment failed: {str(e)}")
        return None

def generate_face_encodings():
    """Generate encodings for all enrolled faces"""
    face_encodings = []
    face_identifiers = []
    
    image_files = get_enrolled_images()
    if not image_files:
        print("No faces found to encode")
        return [], []

    for img_path in image_files:
        try:
            image = face_recognition.load_image_file(str(img_path))
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                face_encodings.append(encodings[0])
                face_identifiers.append(str(img_path.relative_to(ENROLLMENT_DIR)))
                print(f"Generated encoding for: {img_path.name}")
                
        except Exception as e:
            print(f"Failed to encode {img_path}: {str(e)}")
            continue

    if face_encodings:
        # Cache the encodings
        cache_file = CACHE_DIR / "FaceEncodings.p"
        with open(cache_file, 'wb') as f:
            pickle.dump([face_encodings, face_identifiers], f)
        print(f"Successfully encoded {len(face_encodings)} faces")
        
    return face_encodings, face_identifiers

def load_cached_encodings():
    """Load previously cached face encodings"""
    try:
        cache_file = CACHE_DIR / "FaceEncodings.p"
        with open(cache_file, 'rb') as f:
            face_encodings, face_identifiers = pickle.load(f)
        print(f"Loaded {len(face_encodings)} cached encodings")
        return face_encodings, face_identifiers
    except Exception as e:
        print(f"Failed to load cached encodings: {str(e)}")
        return [], []

def process_frame(frame, face_encodings, face_identifiers, awaiting_first_enrollment):
    """Process a video frame for face detection and recognition"""
    display_frame = frame.copy()
    first_face_enrolled = False
    
    # Scale down frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), None, PROCESSING_SCALE, PROCESSING_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    cache_path = os.path.join(CACHE_DIR, "face_groups.json")
    
    for face_encoding, face_location in zip(current_face_encodings, face_locations):
        # Scale coordinates back to original size
        top, right, bottom, left = [int(coord / PROCESSING_SCALE) for coord in face_location]
        scaled_location = (top, right, bottom, left)
        
        if awaiting_first_enrollment:
            name = "First Face - Enrolling..."
            color = (255, 165, 0)  # Orange
            
            if enroll_face(frame, scaled_location):
                face_encodings, face_identifiers = generate_face_encodings()
                first_face_enrolled = True
        else:
            # Check if we have any encodings to compare against
            if face_encodings:
                matches = face_recognition.compare_faces(face_encodings, face_encoding)
                if True in matches:
                    first_match_index = matches.index(True)
                    ID_groups = safe_read_json(cache_path, default_value={})
                    name = find_person_id(ID_groups,str(Path(face_identifiers[first_match_index])))
                    #name = str(Path(face_identifiers[first_match_index])) #.stem
                    color = (0, 255, 0)  # Green
                else:
                    name = "Unknown - Enrolling..."
                    color = (0, 0, 255)  # Red
                    if enroll_face(frame, scaled_location):
                        face_encodings, face_identifiers = generate_face_encodings()
            else:
                # If no encodings exist, treat as unknown and enroll
                name = "Unknown - Enrolling..."
                color = (0, 0, 255)  # Red
                if enroll_face(frame, scaled_location):
                    face_encodings, face_identifiers = generate_face_encodings()
        
        # Draw box and label
        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
        cv2.putText(display_frame, name, (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return display_frame, first_face_enrolled, face_encodings, face_identifiers

def main():
    # Try loading cached encodings first
    face_encodings, face_identifiers = load_cached_encodings()
    
    # If no cache, check enrollment directory
    if not face_encodings:
        face_encodings, face_identifiers = generate_face_encodings()

    awaiting_first_enrollment = not face_encodings
    if awaiting_first_enrollment:
        print("No enrolled faces found. Waiting for first face to enroll...")
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Failed to access camera")
        return
        
    camera.set(3, CAPTURE_WIDTH)
    camera.set(4, CAPTURE_HEIGHT)
    
    print("Face recognition active... Press 'q' to quit")
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
                
            display_frame, first_face_enrolled, face_encodings, face_identifiers = process_frame(
                frame, face_encodings, face_identifiers, awaiting_first_enrollment
            )
            
            if first_face_enrolled:
                awaiting_first_enrollment = False
                print("First face enrolled! Continuing with recognition...")
            
            cv2.imshow("Face Recognition System", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
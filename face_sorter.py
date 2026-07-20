import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gc
import cv2

# Ensure the path is constructed correctly
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Check if the path actually exists
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise IOError("Failed to load Haar cascade file. Check your OpenCV installation.")
st.set_page_config(page_title="Lightweight Face Sorter", layout="wide", initial_sidebar_state="expanded")
st.title("AI Photo Sorter 📸 (Zero-Dependency Version)")

# --- Initialize Classic Computer Vision Tools ---
# 1. Haar Cascade for detecting where the face is
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)

# 2. LBPH Recognizer for identifying who the face belongs to
recognizer = cv2.face.LBPHFaceRecognizer_create()

def process_image_to_gray(file_buffer):
    """Converts uploaded file directly to grayscale numpy array for OpenCV."""
    img = Image.open(file_buffer).convert("L")  # 'L' mode is Grayscale
    
    # Scale down to prevent RAM spikes on massive photos
    img.thumbnail((800, 800), Image.Resampling.BILINEAR)
    return np.array(img, dtype='uint8')

# --- Sidebar: Reference Profiles ---
st.sidebar.header("1. Upload Target Faces")
st.sidebar.write("Upload 1 clear, front-facing photo per person.")
target_files = st.sidebar.file_uploader("Target Faces", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

st.header("2. Upload Photos to Sort")
photos_to_sort = st.file_uploader("Batch Photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

@st.cache_resource
def train_recognizer(uploaded_targets):
    """Trains the LBPH algorithm from scratch based on uploaded targets."""
    faces = []
    ids = []
    name_map = {}
    
    for idx, file in enumerate(uploaded_targets):
        name_map[idx] = file.name.split('.')[0]
        gray_image = process_image_to_gray(file)
        
        # Detect the face
        detected = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
        
        if len(detected) > 0:
            # Take the first face found
            x, y, w, h = detected[0]
            # Crop to just the face
            face_roi = gray_image[y:y+h, x:x+w]
            # Standardize size for the AI model
            face_roi = cv2.resize(face_roi, (200, 200))
            
            faces.append(face_roi)
            ids.append(idx)
            
        del gray_image
        gc.collect()
        
    if len(faces) > 0:
        # Train the model from scratch!
        recognizer.train(faces, np.array(ids))
        return True, name_map
    else:
        return False, {}

# --- Main Logic ---
if target_files and photos_to_sort:
    is_trained, name_dictionary = train_recognizer(target_files)
    
    if not is_trained:
        st.error("Could not find clear faces in the Target images. Try different photos.")
    else:
        st.sidebar.success(f"Trained model on {len(name_dictionary)} targets.")
        
        if st.button("Start Sorting", type="primary"):
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            sorted_photos = {name: [] for name in name_dictionary.values()}
            sorted_photos["Unknown/No Match"] = []
            
            total_photos = len(photos_to_sort)
            
            # Distance threshold (Lower means a stricter match. 70-80 is standard for LBPH)
            MATCH_THRESHOLD = 80  
            
            for i, photo in enumerate(photos_to_sort):
                status_text.text(f"Processing image {i+1} of {total_photos}...")
                
                gray_image = process_image_to_gray(photo)
                detected_faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
                
                matched = False
                for (x, y, w, h) in detected_faces:
                    face_roi = cv2.resize(gray_image[y:y+h, x:x+w], (200, 200))
                    
                    # Predict using our trained model
                    predicted_id, distance = recognizer.predict(face_roi)
                    
                    if distance < MATCH_THRESHOLD:
                        matched_name = name_dictionary[predicted_id]
                        sorted_photos[matched_name].append(photo)
                        matched = True
                        break # Stop looking at other faces in this photo once matched
                
                if not matched:
                    sorted_photos["Unknown/No Match"].append(photo)
                    
                del gray_image
                gc.collect()
                progress_bar.progress((i + 1) / total_photos)
                
            status_text.success("Sorting Complete!")
            
            # Render Results
            for name, photos in sorted_photos.items():
                if photos:
                    st.subheader(f"{name} ({len(photos)} photos)")
                    cols = st.columns(min(len(photos), 5))
                    
                    for idx, p in enumerate(photos[:5]): 
                        p.seek(0)
                        cols[idx].image(p, use_container_width=True)
                    
                    if len(photos) > 5:
                        st.write(f"*...and {len(photos) - 5} more*")

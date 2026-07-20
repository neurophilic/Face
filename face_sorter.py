import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gc
import os
import urllib.request

# --- 1. Application Config ---
st.set_page_config(page_title="AI Photo Sorter", layout="wide")
st.title("AI Photo Sorter 📸")
st.markdown("A lightweight, zero-deep-learning face sorter optimized for cloud deployment.")

# --- 2. Guaranteed Model Initialization ---
@st.cache_resource
def load_cascade():
    """Ensures the Haar Cascade XML is present, downloading it if necessary."""
    cascade_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_filename):
        # Download directly from OpenCV's official GitHub if missing
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, cascade_filename)
        except Exception as e:
            st.error(f"Failed to download cascade file: {e}")
            st.stop()
    return cascade_filename

# Initialize Detector
cascade_path = load_cascade()
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("System Failure: Cascade classifier could not initialize.")
    st.stop()

# Initialize Recognizer (with safety catch for wrong pip packages)
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    st.error("System Failure: The cv2.face module is missing. Check your requirements.txt. You must use 'opencv-contrib-python-headless' and remove any other opencv packages.")
    st.stop()

# --- 3. Image Processing Helpers ---
def process_image_to_gray(file_buffer):
    """Safely converts an uploaded Streamlit file into an OpenCV-ready uint8 grayscale array."""
    img = Image.open(file_buffer).convert("L")
    img.thumbnail((600, 600), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)

def extract_and_sanitize_face(gray_image, x, y, w, h):
    """Extracts the face region and perfectly formats it for the LBPH algorithm."""
    roi = gray_image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
    return roi.astype(np.uint8)

# --- 4. Sidebar & UI ---
st.sidebar.header("1. Upload Targets")
target_files = st.sidebar.file_uploader("Upload 1 clear face per person", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

st.header("2. Upload Batch")
photos_to_sort = st.file_uploader("Upload photos to sort", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# --- 5. Training Phase ---
@st.cache_resource
def train_model(uploaded_files):
    faces = []
    ids = []
    name_map = {}
    
    for idx, file in enumerate(uploaded_files):
        name_map[idx] = file.name.split('.')[0]
        gray = process_image_to_gray(file)
        
        # Detect face
        detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(detected) > 0:
            x, y, w, h = detected[0] # Take the first face found
            face_roi = extract_and_sanitize_face(gray, x, y, w, h)
            faces.append(face_roi)
            ids.append(idx)
            
        del gray
        gc.collect()
        
    if faces:
        recognizer.train(faces, np.array(ids))
        return name_map
    return None

# --- 6. Execution Phase ---
if target_files and photos_to_sort:
    name_dict = train_model(target_files)
    
    if not name_dict:
        st.error("No valid faces found in target images. Please upload clearer photos.")
    else:
        st.sidebar.success(f"Model trained on {len(name_dict)} target profiles.")
        
        if st.button("Start Sorting", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Setup results dictionary
            results = {name: [] for name in name_dict.values()}
            results["Unknown / No Match"] = []
            total_photos = len(photos_to_sort)
            
            for i, photo in enumerate(photos_to_sort):
                status_text.text(f"Processing {i+1} of {total_photos}...")
                
                gray = process_image_to_gray(photo)
                detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                matched = False
                for (x, y, w, h) in detected:
                    # Isolate and sanitize
                    roi = extract_and_sanitize_face(gray, x, y, w, h)
                    
                    try:
                        # Predict
                        label, distance = recognizer.predict(roi)
                        
                        # Distance threshold (lower = stricter match, 80 is a standard baseline)
                        if distance < 80:
                            matched_name = name_dict[label]
                            results[matched_name].append(photo)
                            matched = True
                            break # Move to next photo once matched
                    except cv2.error:
                        # Silently skip regions that cause internal OpenCV calculation errors
                        continue
                        
                if not matched:
                    results["Unknown / No Match"].append(photo)
                    
                # Clean memory
                del gray
                gc.collect()
                progress_bar.progress((i + 1) / total_photos)
                
            status_text.success("Sorting Complete!")
            
            # --- 7. Render Results ---
            for name, photos in results.items():
                if photos:
                    st.subheader(f"{name} ({len(photos)} photos)")
                    cols = st.columns(min(len(photos), 5))
                    
                    for j, p in enumerate(photos[:5]):
                        p.seek(0)
                        cols[j].image(p, use_container_width=True)
                        
                    if len(photos) > 5:
                        st.caption(f"...and {len(photos) - 5} more.")

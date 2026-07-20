import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gc
import os
import urllib.request
import shutil
import time

# --- 1. Application Config ---
st.set_page_config(page_title="Local Photo Sorter", layout="wide")
st.title("Desktop Photo Sorter 📸")
st.markdown("Scans a local folder on your computer and automatically organizes photos by face.")

# --- 2. Guaranteed Model Initialization ---
@st.cache_resource
def load_cascade():
    cascade_filename = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_filename):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        try:
            urllib.request.urlretrieve(url, cascade_filename)
        except Exception as e:
            st.error(f"Failed to download cascade file: {e}")
            st.stop()
    return cascade_filename

cascade_path = load_cascade()
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("System Failure: Cascade classifier could not initialize.")
    st.stop()

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    st.error("System Failure: The cv2.face module is missing. Check your requirements.txt.")
    st.stop()

# --- 3. Image Processing Helpers ---
def process_image_to_gray(image_path_or_buffer):
    """Handles both uploaded file buffers and local file paths."""
    img = Image.open(image_path_or_buffer).convert("L")
    img.thumbnail((600, 600), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)

def extract_and_sanitize_face(gray_image, x, y, w, h):
    roi = gray_image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
    return roi.astype(np.uint8)

def get_image_files(folder_path):
    """Retrieves all valid images from a local folder."""
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = []
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_files.append(os.path.join(folder_path, file))
    return image_files

# --- 4. Sidebar & UI ---
st.sidebar.header("1. Upload Targets")
target_files = st.sidebar.file_uploader("Upload 1 clear face per person", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Algorithm Settings")
MATCH_THRESHOLD = st.sidebar.slider(
    "Match Strictness", 
    min_value=40, max_value=130, value=80,
    help="Lower = Stricter. Higher = Looser."
)

st.header("2. Target Folder")
# Using a local path input. 
batch_folder_path = st.text_input(
    "Enter the absolute path to the folder containing photos to sort:", 
    placeholder="/Users/alle/Pictures/Batch"
)

# --- 5. Training Phase ---
@st.cache_resource
def train_model(uploaded_files):
    faces, ids, name_map = [], [], {}
    for idx, file in enumerate(uploaded_files):
        name_map[idx] = file.name.split('.')[0]
        gray = process_image_to_gray(file)
        detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        if len(detected) > 0:
            x, y, w, h = detected[0] 
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
if target_files and batch_folder_path:
    if not os.path.isdir(batch_folder_path):
        st.warning("Please enter a valid directory path.")
    else:
        photos_to_sort = get_image_files(batch_folder_path)
        
        if not photos_to_sort:
            st.warning("No JPG or PNG files found in that folder.")
        else:
            st.info(f"Found {len(photos_to_sort)} images to scan.")
            
            name_dict = train_model(target_files)
            
            if not name_dict:
                st.error("No valid faces found in target images.")
            else:
                st.sidebar.success(f"Model trained on {len(name_dict)} profiles.")
                
                if st.button("Start Scanning & Sorting", type="primary", use_container_width=True):
                    start_time = time.time()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create the master output directory
                    output_base_dir = os.path.join(batch_folder_path, "Sorted_Photos")
                    os.makedirs(output_base_dir, exist_ok=True)
                    
                    results = {name: [] for name in name_dict.values()}
                    results["Unknown_No_Match"] = []
                    total_photos = len(photos_to_sort)
                    
                    for i, photo_path in enumerate(photos_to_sort):
                        status_text.text(f"Scanning photo {i+1} of {total_photos}: {os.path.basename(photo_path)}")
                        
                        gray = process_image_to_gray(photo_path)
                        detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                        
                        matched = False
                        for (x, y, w, h) in detected:
                            roi = extract_and_sanitize_face(gray, x, y, w, h)
                            try:
                                label, distance = recognizer.predict(roi)
                                if distance < MATCH_THRESHOLD:
                                    matched_name = name_dict[label]
                                    results[matched_name].append(photo_path)
                                    matched = True
                                    
                                    # Copy file to destination folder
                                    dest_folder = os.path.join(output_base_dir, matched_name)
                                    os.makedirs(dest_folder, exist_ok=True)
                                    shutil.copy(photo_path, dest_folder)
                                    break 
                            except cv2.error:
                                continue
                                
                        if not matched:
                            results["Unknown_No_Match"].append(photo_path)
                            dest_folder = os.path.join(output_base_dir, "Unknown_No_Match")
                            os.makedirs(dest_folder, exist_ok=True)
                            shutil.copy(photo_path, dest_folder)
                            
                        del gray
                        gc.collect()
                        progress_bar.progress((i + 1) / total_photos)
                    
                    elapsed_time = time.time() - start_time
                    status_text.empty()
                    progress_bar.empty()
                    
                    # --- 7. Metrics ---
                    st.success(f"✅ Sorting Complete! Files saved to: `{output_base_dir}`")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Processed", total_photos)
                    m2.metric("Processing Time", f"{elapsed_time:.1f} sec")
                    
                    match_rate = ((total_photos - len(results["Unknown_No_Match"])) / total_photos) * 100
                    m3.metric("Match Rate", f"{match_rate:.0f}%")

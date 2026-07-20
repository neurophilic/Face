import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gc
import os

# --- Page Config ---
st.set_page_config(page_title="Lightweight Face Sorter", layout="wide")
st.title("AI Photo Sorter 📸")

# --- Initialize Cascade ---
# Using the local file we bundled in the folder
CASCADE_FILE = "haarcascade_frontalface_default.xml"

if not os.path.exists(CASCADE_FILE):
    st.error(f"Missing {CASCADE_FILE} in your folder. Please upload it to your repo.")
    st.stop()

face_detector = cv2.CascadeClassifier(CASCADE_FILE)
# LBPH recognizer (requires opencv-contrib-python-headless)
recognizer = cv2.face.LBPHFaceRecognizer_create()

def process_image_to_gray(file_buffer):
    img = Image.open(file_buffer).convert("L")
    img.thumbnail((600, 600), Image.Resampling.BILINEAR)
    return np.array(img, dtype='uint8')

# --- Sidebar Inputs ---
st.sidebar.header("1. Upload Target Faces")
target_files = st.sidebar.file_uploader("Upload 1 photo per person", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

st.header("2. Upload Batch")
photos_to_sort = st.file_uploader("Upload photos to sort", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

@st.cache_resource
def train_model(uploaded_files):
    faces, ids, name_map = [], [], {}
    for idx, file in enumerate(uploaded_files):
        name_map[idx] = file.name.split('.')[0]
        gray = process_image_to_gray(file)
        detected = face_detector.detectMultiScale(gray, 1.1, 5)
        if len(detected) > 0:
            x, y, w, h = detected[0]
            face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            faces.append(face_roi)
            ids.append(idx)
    if faces:
        recognizer.train(faces, np.array(ids))
        return name_map
    return None

if target_files and photos_to_sort:
    name_dict = train_model(target_files)
    
    if name_dict and st.button("Start Sorting"):
        results = {name: [] for name in name_dict.values()}
        results["Unknown"] = []
        
        progress = st.progress(0)
        for i, photo in enumerate(photos_to_sort):
            gray = process_image_to_gray(photo)
            detected = face_detector.detectMultiScale(gray, 1.1, 5)
            
            matched = False
            for (x, y, w, h) in detected:
                roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                label, dist = recognizer.predict(roi)
                
                if dist < 80: # Threshold
                    results[name_dict[label]].append(photo)
                    matched = True
                    break
            
            if not matched: results["Unknown"].append(photo)
            progress.progress((i + 1) / len(photos_to_sort))
            
        # Display
        for name, photos in results.items():
            if photos:
                st.subheader(f"{name} ({len(photos)})")
                cols = st.columns(min(len(photos), 5))
                for j, p in enumerate(photos[:5]):
                    p.seek(0)
                    cols[j].image(p, use_container_width=True)

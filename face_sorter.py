import streamlit as st
import cv2
import numpy as np
from PIL import Image
import gc
import os
import urllib.request
import shutil
import time
import tkinter as tk
from tkinter import filedialog

# --- 1. Application Config ---
st.set_page_config(page_title="Auto Face Discovery", layout="wide")
st.title("Auto Face Discovery & Sorter 🤖")
st.markdown("Select a folder. The AI will automatically find, group, and learn unique faces on its own.")

# Initialize session state for the folder path
if "folder_path" not in st.session_state:
    st.session_state.folder_path = ""

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

# --- 3. Image Processing Helpers ---
def process_image_to_gray(image_path):
    img = Image.open(image_path).convert("L")
    img.thumbnail((600, 600), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)

def extract_and_sanitize_face(gray_image, x, y, w, h):
    roi = gray_image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
    return roi.astype(np.uint8)

def get_image_files(folder_path):
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = []
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_files.append(os.path.join(folder_path, file))
    return image_files

# --- 4. Sidebar & UI ---
st.sidebar.header("⚙️ Algorithm Settings")
MATCH_THRESHOLD = st.sidebar.slider(
    "Match Strictness", 
    min_value=40, max_value=120, value=75,
    help="Lower = Stricter (creates more folders). Higher = Looser (groups more faces together)."
)
st.sidebar.caption("Because the AI is discovering faces blindly, adjusting this slider is key to getting perfect groupings.")

# --- Folder Picker Logic ---
st.header("1. Choose Directory")

# The Native Folder Picker Button
if st.button("➕ Select Folder to Scan", type="primary"):
    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Force window to the front
    
    # Open the native OS folder picker
    selected_folder = filedialog.askdirectory(master=root, title="Select Folder Containing Photos")
    root.destroy()
    
    if selected_folder:
        st.session_state.folder_path = selected_folder

# Display chosen path
if st.session_state.folder_path:
    st.success(f"**Target Directory:** `{st.session_state.folder_path}`")
else:
    st.info("Click the button above to select a folder on your computer.")

# --- 5. Execution Phase (Auto-Discovery) ---
if st.session_state.folder_path and os.path.isdir(st.session_state.folder_path):
    photos_to_sort = get_image_files(st.session_state.folder_path)
    
    if not photos_to_sort:
        st.warning("No JPG or PNG files found in that folder.")
    else:
        st.write(f"Found **{len(photos_to_sort)}** images to scan.")
        
        if st.button("Start Auto-Discovery & Sort", use_container_width=True):
            # Create a fresh recognizer for every new run
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            is_trained = False
            current_person_id = 1
            
            output_base_dir = os.path.join(st.session_state.folder_path, "Auto_Sorted_Faces")
            os.makedirs(output_base_dir, exist_ok=True)
            
            # Dictionary to track which images belong to which Person ID
            results = {"No_Faces_Detected": []} 
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_photos = len(photos_to_sort)
            start_time = time.time()
            
            for i, photo_path in enumerate(photos_to_sort):
                status_text.text(f"Scanning photo {i+1} of {total_photos}...")
                gray = process_image_to_gray(photo_path)
                detected = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                
                if len(detected) == 0:
                    results["No_Faces_Detected"].append(photo_path)
                    dest = os.path.join(output_base_dir, "No_Faces_Detected")
                    os.makedirs(dest, exist_ok=True)
                    shutil.copy(photo_path, dest)
                else:
                    for (x, y, w, h) in detected:
                        roi = extract_and_sanitize_face(gray, x, y, w, h)
                        label_array = np.array([current_person_id], dtype=np.int32)
                        
                        # If the model is completely empty, start by training it on the very first face
                        if not is_trained:
                            recognizer.train([roi], label_array)
                            results[f"Person_{current_person_id}"] = [photo_path]
                            
                            # Copy file
                            dest = os.path.join(output_base_dir, f"Person_{current_person_id}")
                            os.makedirs(dest, exist_ok=True)
                            shutil.copy(photo_path, dest)
                            
                            is_trained = True
                            current_person_id += 1
                        else:
                            try:
                                # Ask the AI: Who is this?
                                predicted_id, distance = recognizer.predict(roi)
                                
                                if distance < MATCH_THRESHOLD:
                                    # It's a match! Update the AI's memory of this person to make it smarter
                                    recognizer.update([roi], np.array([predicted_id], dtype=np.int32))
                                    person_key = f"Person_{predicted_id}"
                                    
                                    if photo_path not in results.get(person_key, []):
                                        if person_key not in results: results[person_key] = []
                                        results[person_key].append(photo_path)
                                        dest = os.path.join(output_base_dir, person_key)
                                        os.makedirs(dest, exist_ok=True)
                                        shutil.copy(photo_path, dest)
                                else:
                                    # New face discovered! Add them to the database
                                    recognizer.update([roi], label_array)
                                    person_key = f"Person_{current_person_id}"
                                    results[person_key] = [photo_path]
                                    
                                    dest = os.path.join(output_base_dir, person_key)
                                    os.makedirs(dest, exist_ok=True)
                                    shutil.copy(photo_path, dest)
                                    
                                    current_person_id += 1
                                    
                            except cv2.error:
                                continue
                                
                del gray
                gc.collect()
                progress_bar.progress((i + 1) / total_photos)
            
            elapsed_time = time.time() - start_time
            status_text.empty()
            progress_bar.empty()
            
            # --- 6. Metrics ---
            st.success(f"✅ Discovery Complete! Faces sorted into `{output_base_dir}`")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Processed", total_photos)
            m2.metric("Processing Time", f"{elapsed_time:.1f} sec")
            
            discovered_faces = current_person_id - 1
            m3.metric("Unique People Discovered", discovered_faces)

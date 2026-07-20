import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import gc

st.set_page_config(page_title="AI Photo Sorter", layout="wide", initial_sidebar_state="expanded")
st.title("AI Photo Sorter 📸")

# --- Performance Helper Function ---
def process_and_resize_image(file_buffer, max_size=800):
    """
    Dramatically reduces memory usage by resizing the image before
    handing it to the heavy AI models.
    """
    img = Image.open(file_buffer)
    # Convert to RGB (fixes issues with transparent PNGs or strange formats)
    img = img.convert("RGB") 
    
    # Resize keeping aspect ratio if the image is too large
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Convert PIL Image back to the numpy array face_recognition expects
    return np.array(img)

# --- Sidebar: Reference Profiles ---
st.sidebar.header("1. Upload Target Faces")
st.sidebar.write("Upload one clear, front-facing photo per person.")
target_files = st.sidebar.file_uploader("Target Faces", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

st.header("2. Upload Photos to Sort")
photos_to_sort = st.file_uploader("Batch Photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Cache the target encodings so they aren't recalculated on every button click
@st.cache_resource
def load_target_encodings(uploaded_targets):
    encodings = []
    names = []
    for file in uploaded_targets:
        # Using the optimized resizer
        rgb_image = process_and_resize_image(file)
        
        # 'hog' model is much lighter on RAM than the default 'cnn'
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        face_enc = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_enc) > 0:
            encodings.append(face_enc[0])
            names.append(file.name.split('.')[0])
            
        # Clean up
        del rgb_image
        gc.collect()
        
    return encodings, names

if target_files and photos_to_sort:
    target_encodings, target_names = load_target_encodings(target_files)
    
    if not target_encodings:
        st.error("No faces were detected in the target images. Please try clearer photos.")
    else:
        st.sidebar.success(f"Loaded {len(target_names)} targets.")
        
        if st.button("Start Sorting", type="primary"):
            # UI Containers
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            sorted_photos = {name: [] for name in target_names}
            sorted_photos["Unknown/No Match"] = []
            
            total_photos = len(photos_to_sort)
            
            # Loop through each uploaded photo
            for i, photo in enumerate(photos_to_sort):
                status_text.text(f"Processing image {i+1} of {total_photos}...")
                
                # Use our memory-saving function
                small_image = process_and_resize_image(photo)
                
                # Find faces in the resized image
                face_locations = face_recognition.face_locations(small_image, model="hog")
                face_encs = face_recognition.face_encodings(small_image, face_locations)
                
                matched = False
                for face_encoding in face_encs:
                    # Compare against known targets
                    matches = face_recognition.compare_faces(target_encodings, face_encoding, tolerance=0.5)
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        matched_name = target_names[first_match_index]
                        sorted_photos[matched_name].append(photo)
                        matched = True
                        break 
                
                if not matched:
                     sorted_photos["Unknown/No Match"].append(photo)
                
                # CRITICAL: Force garbage collection 
                del small_image
                del face_locations
                del face_encs
                gc.collect()
                
                progress_bar.progress((i + 1) / total_photos)
                
            status_text.success("Sorting Complete!")
            
            # Display results efficiently
            for name, photos in sorted_photos.items():
                if photos:
                    st.subheader(f"{name} ({len(photos)} photos)")
                    cols = st.columns(min(len(photos), 5))
                    
                    for idx, p in enumerate(photos[:5]): 
                        p.seek(0) # Reset file pointer
                        # Use Streamlit's built-in fast image renderer
                        cols[idx].image(p, use_container_width=True)
                    
                    if len(photos) > 5:
                        st.write(f"*...and {len(photos) - 5} more*")

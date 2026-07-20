import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import gc
import io

# Setup page configuration
st.set_page_config(page_title="AI Photo Sorter", layout="wide")
st.title("AI Photo Sorter 📸")

# --- Sidebar: Reference Profiles ---
st.sidebar.header("1. Upload Target Faces")
st.sidebar.write("Upload one clear, front-facing photo for each person you want to sort.")
target_files = st.sidebar.file_uploader("Target Faces", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# --- Main Page: Batch Photos ---
st.header("2. Upload Photos to Sort")
photos_to_sort = st.file_uploader("Batch Photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Cache the target encodings to save memory upon Streamlit reruns
@st.cache_resource
def load_target_encodings(uploaded_targets):
    encodings = []
    names = []
    for file in uploaded_targets:
        image = face_recognition.load_image_file(file)
        face_enc = face_recognition.face_encodings(image)
        if len(face_enc) > 0:
            encodings.append(face_enc[0])
            names.append(file.name.split('.')[0]) # Uses filename as the person's name
    return encodings, names

if target_files and photos_to_sort:
    target_encodings, target_names = load_target_encodings(target_files)
    
    if not target_encodings:
        st.error("No faces were detected in the target images. Please try clearer photos.")
    else:
        st.success(f"Loaded {len(target_names)} target profiles into memory.")
        
        if st.button("Start Sorting"):
            st.write("### Sorting Results")
            
            # Initialize a dictionary to hold the sorted output
            sorted_photos = {name: [] for name in target_names}
            sorted_photos["Unknown/No Match"] = []
            
            progress_bar = st.progress(0)
            
            # Loop through each uploaded photo
            for i, photo in enumerate(photos_to_sort):
                # Load the current image
                image = face_recognition.load_image_file(photo)
                face_locations = face_recognition.face_locations(image)
                face_encs = face_recognition.face_encodings(image, face_locations)
                
                matched = False
                for face_encoding in face_encs:
                    # Compare against known targets with a strict tolerance
                    matches = face_recognition.compare_faces(target_encodings, face_encoding, tolerance=0.5)
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        matched_name = target_names[first_match_index]
                        sorted_photos[matched_name].append(photo)
                        matched = True
                        break 
                
                if not matched:
                     sorted_photos["Unknown/No Match"].append(photo)
                
                # CRITICAL: Force garbage collection to prevent cloud RAM overflow limits
                del image
                del face_locations
                del face_encs
                gc.collect()
                
                # Update visual progress
                progress_bar.progress((i + 1) / len(photos_to_sort))
                
            st.success("Sorting Complete!")
            
            # Display the sorted categories
            for name, photos in sorted_photos.items():
                if photos:
                    st.subheader(f"{name} ({len(photos)} photos)")
                    cols = st.columns(min(len(photos), 5))
                    
                    # Display a preview of the sorted photos
                    for idx, p in enumerate(photos[:5]): 
                        p.seek(0)
                        cols[idx].image(Image.open(p), use_container_width=True)
                    if len(photos) > 5:
                        st.write(f"*...and {len(photos) - 5} more*")

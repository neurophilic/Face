import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import gc

st.set_page_config(page_title="AI Photo Sorter", layout="wide", initial_sidebar_state="expanded")
st.title("AI Photo Sorter 📸")

# --- Ultra-Fast Image Loader & Resizer ---
def process_and_resize_image(file_buffer, max_size=600):
    """
    Downsamples image to max 600px on the longest side.
    600px is the sweet spot for rapid face detection without losing accuracy.
    """
    img = Image.open(file_buffer).convert("RGB")
    
    # Fast resize using bilinear interpolation
    img.thumbnail((max_size, max_size), Image.Resampling.BILINEAR)
    return np.asarray(img)

# --- Sidebar: Reference Profiles ---
st.sidebar.header("1. Upload Target Faces")
st.sidebar.write("Upload 1 clear face photo per person.")
target_files = st.sidebar.file_uploader("Target Faces", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

st.header("2. Upload Photos to Sort")
photos_to_sort = st.file_uploader("Batch Photos", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_target_encodings(uploaded_targets):
    encodings = []
    names = []
    for file in uploaded_targets:
        rgb_image = process_and_resize_image(file, max_size=600)
        
        # upsample=0 skips image enlargement (huge speed boost)
        locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=0, model="hog")
        
        if locations:
            face_enc = face_recognition.face_encodings(rgb_image, known_face_locations=locations, num_jitters=1)
            if face_enc:
                encodings.append(face_enc[0])
                names.append(file.name.split('.')[0])
                
        del rgb_image
        gc.collect()
        
    return np.array(encodings) if encodings else None, names

if target_files and photos_to_sort:
    target_encodings, target_names = load_target_encodings(target_files)
    
    if target_encodings is None or len(target_encodings) == 0:
        st.error("No faces were detected in the target images. Please upload clearer photos.")
    else:
        st.sidebar.success(f"Loaded {len(target_names)} target profiles.")
        
        if st.button("Start Sorting", type="primary"):
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            sorted_photos = {name: [] for name in target_names}
            sorted_photos["Unknown/No Match"] = []
            
            total_photos = len(photos_to_sort)
            
            # Match threshold (0.50 is standard; lower = stricter match)
            MATCH_THRESHOLD = 0.50 

            for i, photo in enumerate(photos_to_sort):
                # Throttle progress bar updates to save WebSocket/UI rendering overhead
                if i % max(1, total_photos // 20) == 0 or i == total_photos - 1:
                    status_text.text(f"Processing image {i+1} of {total_photos}...")
                    progress_bar.progress((i + 1) / total_photos)
                
                # Resize image
                small_image = process_and_resize_image(photo, max_size=600)
                
                # 1. Detect locations (upsample=0)
                face_locations = face_recognition.face_locations(small_image, number_of_times_to_upsample=0, model="hog")
                
                # 2. Skip encoding entirely if no faces exist in the photo
                if face_locations:
                    face_encs = face_recognition.face_encodings(small_image, known_face_locations=face_locations, num_jitters=1)
                    
                    matched = False
                    for face_encoding in face_encs:
                        # Vectorized Euclidean Distance calculation (Fast C-Level Math)
                        distances = face_recognition.face_distance(target_encodings, face_encoding)
                        best_match_idx = np.argmin(distances)
                        
                        if distances[best_match_idx] <= MATCH_THRESHOLD:
                            matched_name = target_names[best_match_idx]
                            sorted_photos[matched_name].append(photo)
                            matched = True
                            break # Found the closest target for this photo
                            
                    if not matched:
                        sorted_photos["Unknown/No Match"].append(photo)
                else:
                    sorted_photos["Unknown/No Match"].append(photo)
                
                # Free memory immediately
                del small_image
                del face_locations
                gc.collect()
                
            status_text.success("Sorting Complete!")
            
            # Render Results Grid
            for name, photos in sorted_photos.items():
                if photos:
                    st.subheader(f"{name} ({len(photos)} photos)")
                    cols = st.columns(min(len(photos), 5))
                    
                    for idx, p in enumerate(photos[:5]): 
                        p.seek(0)
                        cols[idx].image(p, use_container_width=True)
                    
                    if len(photos) > 5:
                        st.write(f"*...and {len(photos) - 5} more*")

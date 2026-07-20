import os
import shutil
import face_recognition
import cv2

def load_reference_faces(reference_dir):
    """Loads reference images to learn what each person looks like."""
    known_encodings = []
    known_names = []
    
    for filename in os.listdir(reference_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(reference_dir, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            
            if encodings:
                # Use the filename (without extension) as the person's name
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name.title())
                
    return known_encodings, known_names

def sort_album(album_dir, output_dir, known_encodings, known_names):
    """Scans the album and copies photos into named folders."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(album_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(album_dir, filename)
            image = face_recognition.load_image_file(filepath)
            
            # Find all faces in the current image
            encodings = face_recognition.face_encodings(image)
            found_names = set()

            for encoding in encodings:
                # Compare face to known faces
                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                    found_names.add(name)

            # If no known faces are found, put it in "Unknown"
            if not found_names:
                found_names.add("Unknown")

            # Copy the image to the respective folder(s)
            for name in found_names:
                person_dir = os.path.join(output_dir, name)
                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)
                
                shutil.copy(filepath, os.path.join(person_dir, filename))
                print(f"[SUCCESS] Copied {filename} to {name}'s folder.")

if __name__ == "__main__":
    print("=========================================")
    print("        AI PHOTO ALBUM SORTER            ")
    print("=========================================")
    print("Note: Use absolute paths (e.g., C:/Photos/Album)\n")
    
    ref_dir = input("1. Enter path to 'Reference Faces' folder: ").strip('\"\'')
    album_dir = input("2. Enter path to 'Unsorted Album' folder: ").strip('\"\'')
    out_dir = input("3. Enter path for the 'Sorted Output' folder: ").strip('\"\'')

    print("\n[INFO] Loading reference faces...")
    encodings, names = load_reference_faces(ref_dir)
    print(f"[INFO] Learned {len(names)} faces: {', '.join(names)}")

    print("\n[INFO] Scanning and sorting album...")
    sort_album(album_dir, out_dir, encodings, names)
    
    print("\n[DONE] Sorting complete!")
    input("Press Enter to exit.")
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Function to load JPG images and their labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in tqdm(os.listdir(folder), desc="Loading images"):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                # Extract label from filename
                label = filename.split("(")[1].split(")")[0]
                labels.append(label)
                print(f"Image: {filename}, Label: {label}")
    return images, labels

# Function to extract face embeddings using a pre-trained model
def extract_embeddings(images, target_size=(100, 100)):
    embeddings = []
    for image in tqdm(images, desc="Extracting embeddings"):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Ensure only one face is detected per image
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            
            # Crop and resize face region to target size
            face_roi_resized = cv2.resize(gray[y:y+h, x:x+w], target_size)
            
            # Flatten the face embedding
            flattened_embedding = face_roi_resized.flatten()
            embeddings.append(flattened_embedding)
    
    return embeddings

# Main function to train the model
def train_model(images_folder, model_output_file):
    images, labels = load_images_from_folder(images_folder)
    if len(images) != len(labels):
        raise ValueError("Number of images and labels do not match.")
    
    face_embeddings = extract_embeddings(images)
    
    # Convert labels into numerical form
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Train Support Vector Machine (SVM) classifier with adjusted threshold
    svm = SVC(C=1.0, kernel='linear', probability=True)
    svm.fit(np.array(face_embeddings), labels_encoded)
    
    # Save trained model to file
    with open(model_output_file, 'wb') as f:
        np.save(f, label_encoder.classes_)
        np.save(f, svm)

# Example usage with the provided folder location:
train_model(r"D:\MY WEBSITES\Drowsiness\face", 'trained_model12.npy')

import cv2
import numpy as np

# Function to load the trained model
def load_trained_model(model_file):
    with open(model_file, 'rb') as f:
        classes = np.load(f, allow_pickle=True)
        svm = np.load(f, allow_pickle=True).item()  # Load as dictionary
    return classes, svm

# Function to recognize faces using the trained model
# Function to recognize faces using the trained model
def recognize_faces(image, model_file):
    # Load trained model
    classes, svm = load_trained_model(model_file)
    
    # Preprocess image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Extract face region
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return "No face detected", None
    
    (x, y, w, h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    
    # Resize face region to match dimensions expected by SVM classifier
    face_roi_resized = cv2.resize(face_roi, (100, 100))
    
    # Predict using SVM classifier
    prediction = svm.predict(face_roi_resized.flatten().reshape(1, -1))
    probability = np.max(svm.decision_function(face_roi_resized.flatten().reshape(1, -1)))
    label = classes[prediction[0]]
    
    if probability > 0.9:  # Adjust threshold as needed
        return f"Known person: {label}", faces[0]
    else:
        return "Unknown person", faces[0]


# Open webcam
cap = cv2.VideoCapture(0)

# Load the trained model
model_file = 'trained_model12.npy'

while True:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        break
    
    # Recognize faces in the frame
    result, face_coords = recognize_faces(frame, model_file)
    
    # Draw bounding box around the face
    if face_coords is not None:
        (x, y, w, h) = face_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display recognition result
    cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow('Face Recognition', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

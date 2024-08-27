import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import winsound  # For generating beep sound
from twilio.rest import Client  # For sending SMS alerts

# Load the pre-trained face detection model
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
model = keras.models.load_model(r"C:\Users\SomeshP\Downloads\converted_keras (1)\keras_model.h5")

# Define a function to map class indices to class names
def get_class_name(class_no):
    class_names = ["somesh", "unknown", "chinmay"]
    return class_names[class_no]

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width
cap.set(4, 480)  # Set the height

# Font settings for displaying text on the video
font = cv2.FONT_HERSHEY_COMPLEX

# Initialize flag for alert
alert_sent = False

# Twilio credentials
account_sid = 'AC2ea4e59b651a513a324714925bc1fbb6'
auth_token = 'a965ffaa9ff334742a239225c80dd8b0'
client = Client(account_sid, auth_token)

# Function to send alert message to phone
def send_alert():
    message = client.messages.create(
                    body="Unknown face detected! Please check the security camera.",
                    from_='+15074364225',
                    to='+919111269576'
                )

while True:
    
    success, img_original = cap.read()
    
    
    faces = facedetect.detectMultiScale(img_original, 1.3, 5)
    
    
    for x, y, w, h in faces:
        # Extract the face region
        crop_img = img_original[y:y+h, x:x+w]
        
        
        img = cv2.resize(crop_img, (224, 224))
        img = img / 255.0  
        img = np.expand_dims(img, axis=0)  
        
    
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        probability_value = np.amax(prediction)
        
        
        cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        
        cv2.putText(img_original, str(get_class_name(class_index)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_original, str(round(probability_value * 100, 2)) + "%", (x, y+h+25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        
        if get_class_name(class_index) == "unknown":
            if not alert_sent:
                
                winsound.Beep(1000, 2000)  
                
                send_alert()
                alert_sent = True
        else:
            alert_sent = False

    
    cv2.imshow("Face Recognition", img_original)
    
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

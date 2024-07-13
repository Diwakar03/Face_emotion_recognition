import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
model_path = 'models/facial_expression_model.h5'
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    reshaped = np.reshape(resized, (1, 48, 48, 1))
    return reshaped / 255.0

# Function to detect expression based on the new model's output format
def detect_expression(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    max_index = np.argmax(prediction)
    return ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'][max_index]

# Function to detect faces and add bounding boxes with expression labels
def detect_faces_and_expressions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for idx, (x, y, w, h) in enumerate(faces):
        face = frame[y:y+h, x:x+w]
        expression = detect_expression(face)
        person_id = f'P{idx+1}'
        
        color = {
            'angry': (0, 0, 255),
            'disgust': (0, 255, 255),
            'fear': (128, 0, 128),
            'happy': (0, 255, 0),
            'neutral': (255, 0, 0),
            'sad': (255, 255, 0),
            'surprise': (255, 165, 0)
        }[expression]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f'{person_id}: {expression}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# Custom HTML for larger checkbox
st.markdown("""
<style>
.stCheckbox > div {
  font-size: 18px;
}
.stCheckbox div > label {
  font-size: 20px;
  padding: 10px;
  border-radius: 5px;
  background-color: #f0f0f0;
  cursor: pointer;
}
.stCheckbox input[type="checkbox"] {
  width: 20px;
  height: 20px;
}
</style>
""", unsafe_allow_html=True)

# Streamlit app
st.title('Face Emotion Recognition')
st.write("Upload a video file or use your webcam to detect facial expressions")

# Video file upload option
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

# Webcam option with custom checkbox styling
use_webcam = st.checkbox("Use webcam")

# Display the video or webcam feed
if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    cap = cv2.VideoCapture("temp_video.mp4")
    
    if not cap.isOpened():
        st.error("Error opening video stream or file")
elif use_webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error opening webcam")
else:
    st.write("Please upload a video file or select the webcam option.")

# Stop button
stop = st.button("Stop Capturing")

# Optimize frame rate and processing
FRAME_SKIP = 5  # Process every 5th frame
FRAME_WIDTH = 640  # Resize frame width
FRAME_HEIGHT = 480  # Resize frame height

if uploaded_file is not None or use_webcam:
    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Detect faces and expressions
        frame_with_detections = detect_faces_and_expressions(frame)

        # Display the frame in Streamlit
        stframe.image(frame_with_detections, channels="BGR")

    cap.release()

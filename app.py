# Python In-built packages
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
# st.set_page_config(
#     page_title="Mang Hijau",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# Main page heading
st.title("Catch the Trash")

# Sidebar
st.sidebar.header("Klasifikasi")

# Model Options
model_type = 'Detection'

# Set the model confidence to a static value of 60%
confidence = 0.60

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Automatically perform detection using webcam
# Create a container for the webcam stream
video_container = st.empty()

# Create a container in the sidebar for classifications
classification_container = st.sidebar.empty()

def classify_object(label_tensor):
    """Classify the object as 'Organik' or 'Non Organik'."""
    # Convert the tensor to a label string
    label = label_tensor.item()  # Get the value from the tensor

    # Assume label mapping (you may need to adjust based on your model's labels)
    if label == 0:  # Assuming 0 corresponds to "Botol_Plastik"
        return "Non Organik"
    elif label == 1:  # Assuming 1 corresponds to "Kaleng"
        return "Non Organik"
    elif label == 2:  # Assuming 2 corresponds to "Kertas"
        return "Organik"
    else:
        return "Organik"

def process_frame(frame):
    """Process the frame by resizing and performing object detection."""
    # Resize frame to speed up processing
    small_frame = cv2.resize(frame, (640, 480))
    # Perform detection on the frame
    results = model.predict(small_frame, conf=confidence)
    boxes = results[0].boxes
    frame_with_boxes = results[0].plot()[:, :, ::-1]  # Convert from BGR to RGB
    return frame_with_boxes, boxes

def play_webcam(confidence, model):
    """Capture and process video from the webcam."""
    cap = cv2.VideoCapture(0)  # 0 is typically the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        # Process the frame and update the display
        frame_with_boxes, boxes = process_frame(frame)
        video_container.image(frame_with_boxes, channels='RGB', use_column_width=True)

        # Check if any objects are detected
        if len(boxes) > 0:
            classifications = []
            for box in boxes:
                label_tensor = box.cls  # Get the label tensor
                classification = classify_object(label_tensor)
                classifications.append(classification)

                # Save the detected frame as an image
                image = Image.fromarray(frame_with_boxes)
                image.save(f"{classification}.jpg")

            # Display the classifications in the sidebar
            classification_container.markdown("### Klasifikasi:")
            for classification in classifications:
                classification_container.markdown(f"- {classification}")

        # Optionally, you can add logic here to break the loop if needed

    cap.release()

# Start processing the webcam feed
play_webcam(confidence, model)

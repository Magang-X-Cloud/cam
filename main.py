import streamlit as st
from streamlit_option_menu import option_menu
from create_something import main as creative 
from detection import detect_faces
# from app import play_webcam, confidence, model
import settings
import helper
# Python In-built packages
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

with st.sidebar:
    selected = option_menu("Main Menu", ["Detection", 'Creation', ], 
        icons=['house', 'gear'], menu_icon="cast", default_index=0)

if selected == 'Detection':
    st.title("Catch the Trash")
    model_type = 'Detection'
    confidence = 0.60
    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)
    video_container = st.empty()


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
                    # classification = classify_object(label_tensor)
                    # classifications.append(classification)

                    

        cap.release()

    # Start processing the webcam feed
    play_webcam(confidence, model)
        
    # creative()
elif selected == 'Creation':
    # st.write(st.secrets['QWEN_API']['APP_ID'])
    st.title("Lets Create Something")
    creative()
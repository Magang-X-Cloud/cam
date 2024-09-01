import streamlit as st
import cv2
import settings

def detect_faces():
    
#     confidence = float(st.sidebar.slider(
#     "Select Model Confidence", 25, 100, 40)) / 100
#     try:
#         model = helper.load_model(model_path)
#     except Exception as ex:
#         st.error(f"Unable to load model. Check the specified path: {model_path}")
#         st.error(ex)
#     st.sidebar.header("Image/Video Config")
#     source_radio = st.sidebar.radio(
#     "Select Source", settings.SOURCES_LIST)

#     source_img = None
# # If image is selected
#     if source_radio == settings.IMAGE:
#         source_img = st.sidebar.file_uploader(
#         "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame,channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    cap.release()
    cv2.destroyAllWindows()


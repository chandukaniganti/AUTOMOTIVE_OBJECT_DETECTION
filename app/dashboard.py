import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="Automotive Object Detection", layout='wide')
st.title("🚘 Automotive Object Detection — YOLOv8")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# File uploader
uploaded = st.file_uploader("Upload a road video (mp4)", type=['mp4', 'mov', 'avi'])

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    conf = st.slider('Confidence threshold', 0.0, 1.0, 0.25)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf)[0]
        annotated = results.plot()

        # Convert BGR to RGB for Streamlit display
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated, channels='RGB')

    cap.release()
else:
    st.info('Upload a video to start the demo.')

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:45:18 2024

@author: shan2
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load the YOLOv8 model
model = YOLO(r"C:/Users/Dell/Downloads/best.pt")  # Replace with your trained model path

# Streamlit App
st.title("YOLOv8 Object Detection App")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        # Process image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Running inference...")

        # Convert to OpenCV format
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Perform inference
        results = model.predict(image_bgr)

        # Display results
        results_img = results[0].plot()  # Use plot method to visualize results
        st.image(results_img, caption='Detected Objects', use_column_width=True)

    elif "video" in file_type:
        # Save uploaded video to a temporary file
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Read video using OpenCV
        cap = cv2.VideoCapture(temp_video_path)
        stframe = st.empty()  # Streamlit frame for video output

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop when video ends

            # Perform inference on the frame
            results = model.predict(frame)

            # Draw detections on the frame
            output_frame = results[0].plot()

            # Convert BGR to RGB for Streamlit display
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame in Streamlit
            stframe.image(output_frame, channels="RGB", use_column_width=True)

        cap.release()
        st.write("Video processing complete.")
# python -m streamlit run d:/Parking_images/app_3.py
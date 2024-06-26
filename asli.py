import streamlit as st
import logging
import os
import tempfile
import av
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)
from ultralytics import YOLO
import supervision as sv

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logging.basicConfig(level=logging.WARNING)
st.set_page_config(page_title="YOLOvWeld", page_icon="🤖")

# Define the zone polygon
zone_polygon_m = np.array([[160, 100], 
                           [160, 380], 
                           [481, 380], 
                           [481, 100]], dtype=np.int32)

# Initialize the YOLOv8 model
@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

# Load the YOLO model (this will be cached)
model = load_yolo_model("best.pt")  # Ganti "best.pt" dengan nama model Anda

# Initialize the tracker, annotators and zone
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
zone = sv.PolygonZone(polygon=zone_polygon_m, frame_resolution_wh=(642, 642))

zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.red(),
    thickness=2,
    text_thickness=4,
    text_scale=2
)

def draw_annotations(frame, boxes, masks, names):
    for box, name in zip(boxes, names):
        color = (0, 255, 0)  # Green color for bounding boxes

        # Draw bounding box
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

        # Check if masks are available
        if masks is not None:
            mask = masks[frame_number]
            alpha = 0.3  # Transparency of masks

            # Draw mask
            frame[mask > 0] = frame[mask > 0] * (1 - alpha) + np.array(color) * alpha

        # Display class name
        cv2.putText(frame, name, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def main():
    st.title("🤖 YOLOvWeld")
    st.subheader("YOLOv8 & Streamlit Web Integration")
    st.sidebar.title("Select an option ⤵️")
    choice = st.sidebar.radio("", ("Capture Image And Predict", ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️"),
                            index = 1)
    conf = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)
        
    if  choice == "Capture Image And Predict":
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            results = model.predict(cv2_img)

            if isinstance(results, list):
                results1 = results[0]  
            else:
                results1 = results
    
            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > conf]
            labels = [
                f"#{index + 1}: {results1.names[class_id]}"
                for index, class_id in enumerate(detections.class_id)
            ]

            labels1 = [
                f"#{index + 1}: {results1.names[class_id]} (Accuracy: {detections.confidence[index]:.2f})"
                for index, class_id in enumerate(detections.class_id)
            ]

            annotated_frame1 = box_annotator.annotate(cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
            count_text = f"Objects in Frame: {len(detections)}" 
            cv2.putText(annotated_frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            annotated_frame = av.VideoFrame.from_ndarray(annotated_frame1, format="bgr24")
            st.image(annotated_frame.to_ndarray(), channels="BGR")
            st.write(':orange[ Info : ⤵️ ]')
            st.json(labels1)
            st.subheader("", divider='rainbow')

    elif choice == ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️":
        uploaded_files = st.file_uploader("Choose images", type=['png', 'jpg', 'webp', 'bmp'], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            bytes_data = uploaded_file.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            results = model.predict(cv2_img)

            if isinstance(results, list):
                results1 = results[0]  
            else:
                results1 = results
    
            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > conf]
            labels = [
                f"#{index + 1}: {results1.names[class_id]}"
                for index, class_id in enumerate(detections.class_id)
            ]

            labels1 = [
                f"#{index + 1}: {results1.names[class_id]} (Accuracy: {detections.confidence[index]:.2f})"
                for index, class_id in enumerate(detections.class_id)
            ]

            annotated_frame1 = box_annotator.annotate(cv2_img, detections=detections)
            annotated_frame1 = label_annotator.annotate(annotated_frame1, detections=detections, labels=labels)
            count_text = f"Objects in Frame: {len(detections)}" 
            cv2.putText(annotated_frame1, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            annotated_frame = av.VideoFrame.from_ndarray(annotated_frame1, format="bgr24")
            st.image(annotated_frame.to_ndarray(), channels="BGR")
            st.write(':orange[ Info : ⤵️ ]')
            st.json(labels1)
            st.subheader("", divider='rainbow')

if __name__ == '__main__':
    main()

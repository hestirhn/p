import streamlit as st
import logging
import os
import av
import cv2
import numpy as np
import json
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
st.set_page_config(page_title="Ai Object Detection", page_icon="🤖")

# Define the zone polygon
zone_polygon_m = np.array([[160, 100], 
                           [160, 380], 
                           [481, 380], 
                           [481, 100]], dtype=np.int32)

# Initialize the YOLOv8 model
@st.cache
def load_yolo_model(model_path):
    return YOLO(model_path)

# Load the YOLO model (this will be cached)
model = load_yolo_model("best.pt")

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

def save_detections_to_json(file_path, image_path, detections, score_threshold):
    detections_data = [{
        "box": box.tolist(),
        "confidence": float(conf),
        "class_id": int(class_id)
    } for box, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id)]
    
    info = {
        "image_path": image_path,
        "score_threshold": score_threshold,
        "detections": detections_data
    }
    
    with open(file_path, 'w') as f:
        json.dump(info, f)

def main():
    st.title("🤖 Ai Object Detection")
    st.subheader("YOLOv8 & Streamlit WebRTC Integration :)")
    st.sidebar.title("Select an option ⤵️")
    choice = st.sidebar.radio("", ("Capture Image And Predict", ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️", "Upload Video"), index=1)
    conf = st.slider("Score threshold", 0.0, 1.0, 0.3, 0.05)
        
    if choice == "Capture Image And Predict":
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
            st.json(labels)
            st.subheader("", divider='rainbow')

            # Create download buttons for image and JSON
            if st.button("Save Image with Detections"):
                save_path = "detected_image.jpg"
                cv2.imwrite(save_path, annotated_frame1)
                st.success(f"Image with detections saved as {save_path}")
            
                # Create download link for image
                with open(save_path, "rb") as f:
                    image_bytes = f.read()
                st.download_button(
                    label="Download Image with Detections (JPG)",
                    data=image_bytes,
                    file_name="detected_image.jpg",
                    mime="image/jpeg"
                )
                st.download_button(
                    label="Download Detections Info",
                    data=json.dumps(data_to_download),
                    file_name="detections_info.json",
                    mime="application/json"
                )
            
    elif choice == ":rainbow[Multiple Images Upload -]🖼️🖼️🖼️":
        uploaded_files = st.file_uploader("Choose images", type=['png', 'jpg', 'webp', 'bmp'], accept_multiple_files=True)
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            results = model.predict(cv2_img)

            if isinstance(results, list):
                results1 = results[0]  
            else:
                results1 = results
    
            detections = sv.Detections.from_ultralytics(results1)
            detections = detections[detections.confidence > conf]
            labels = [
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
            st.json(labels)
            st.subheader("", divider='rainbow')

            # Create download buttons for image and JSON
            if st.button("Save Image with Detections"):
                save_path = "detected_image.jpg"
                cv2.imwrite(save_path, annotated_frame1)
                st.success(f"Image with detections saved as {save_path}")
            
                # Create download link for image
                with open(save_path, "rb") as f:
                    image_bytes = f.read()
                st.download_button(
                    label="Download Image with Detections (JPG)",
                    data=image_bytes,
                    file_name="detected_image.jpg",
                    mime="image/jpeg"
                )
                st.download_button(
                    label="Download Detections Info",
                    data=json.dumps(data_to_download),
                    file_name="detections_info.json",
                    mime="application/json"
                )

if __name__ == '__main__':
    main()

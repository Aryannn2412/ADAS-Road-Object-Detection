import streamlit as st
import cv2
import os
import random
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import queue

# Define paths
model_path = '/Users/aryanjha/Downloads/MAJOR PROJEC- ADAS/detect/train2/weights/best.pt'
image_folder = '/Users/aryanjha/Downloads/MAJOR PROJEC- ADAS/road_object_detection/images/train'
upload_folder = '/Users/aryanjha/Downloads/MAJOR PROJEC- ADAS/uploads'

# Load models
custom_model = YOLO(model_path)
video_model = YOLO("yolov8n.pt")

st.set_page_config(page_title="ADAS Road Object Detection", layout="wide")
st.title("🚘 ADAS Road Object Detection System")

# --- CSS Styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;600&display=swap');
        html, body, .stApp {
            font-family: 'Poppins', sans-serif;
            background-image: linear-gradient(rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0.85)),
                              url('https://images.unsplash.com/photo-1610394219790-5e4b27f8d7b0?auto=format&fit=crop&w=1920&q=80');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: white;
        }
        .block-container {
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2rem;
            margin: 2rem;
            border-radius: 20px;
            box-shadow: 0 0 20px rgba(255, 204, 0, 0.35);
        }
        h1, h2, h3, h4 {
            color: #ffcc00;
            font-weight: 600;
            text-shadow: 1px 1px 2px black;
        }
        .stButton button {
            background-color: #ffcc00;
            color: black;
            border-radius: 12px;
            padding: 0.6rem 1.4rem;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 0 10px #ffcc00aa;
        }
        .stButton button:hover {
            background-color: #e6b800;
            transform: scale(1.05);
            box-shadow: 0 0 15px #ffcc00;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #000000cc;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            z-index: 999;
        }
        .logo {
            width: 60px;
            height: 60px;
        }
        .element-container:has(.chatbot), .element-container:has(.sos) {
            transition: all 0.3s ease-in-out;
        }
        .chatbot:hover, .sos:hover {
            filter: brightness(120%);
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/684/684908.png", width=80)
st.sidebar.title("Upload Input")

input_type = st.sidebar.radio("Choose input type:", ["Image", "Random Train Image", "Video", "Webcam", "SOS Alert"])

# Tabs Setup
tabs = st.tabs(["📊 Dashboard", "🚦 Detection", "🤖 Chatbot"])

# Session state for stats
if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_objects": 0,
        "object_counts": {},
        "alerts_triggered": 0,
        "webcam_usage": 0,
        "video_runs": 0,
        "sos_activations": 0,
        "last_alert": ""
    }

# Detection responses
detection_responses = {
    "person": "🚶 Pedestrian detected! Slow down.",
    "car": "🚗 Vehicle ahead! Maintain safe distance.",
    "dog": "🐶 Animal on road! Stop or slow down.",
    "cat": "🐱 Animal detected! Proceed with caution.",
    "bicycle": "🚲 Cyclist nearby! Watch your speed.",
    "truck": "🚚 Large vehicle detected! Keep distance."
}

def sos_alert_ui():
    st.subheader("🚨 SOS Alert Mode")
    st.markdown("This mode can be used in case of emergencies. It sends alert signals to the emergency contact system.")
    st.warning("⚠️ You have activated SOS Mode. Immediate action will be taken.")
    if st.button("Send SOS Signal"):
        st.session_state.stats["sos_activations"] += 1
        st.success("✅ SOS Signal Sent to Emergency Contacts!")

with tabs[0]:
    st.header("📊 Detection Dashboard")
    stats = st.session_state.stats
    most_frequent = max(stats["object_counts"].items(), key=lambda x: x[1], default=("None", 0))[0]
    avg_conf = float(sum([conf for conf in stats["object_counts"].values()]) / len(stats["object_counts"]) if stats["object_counts"] else 0)
    st.markdown(f"""
        - Total Objects Detected: {stats['total_objects']}
        - Most Frequent Object: {most_frequent}
        - Average Confidence: {avg_conf:.2f}%
        - Alerts Triggered: {stats['alerts_triggered']}
        - Webcam Usage Count: {stats['webcam_usage']}
        - Video Detections Run: {stats['video_runs']}
        - SOS Activations: {stats['sos_activations']}
    """)

with tabs[1]:
    if input_type == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            temp_path = os.path.join(upload_folder, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(temp_path, caption="Uploaded Image", use_container_width=True)
            if st.button("🚀 Detect"):
                results = custom_model(temp_path)[0]
                plotted = results.plot()
                confs = results.boxes.conf
                avg_conf = float(confs.mean()) * 100 if confs is not None else 0.0
                st.image(plotted, caption=f"🔍 Detection Result (Confidence: {avg_conf:.2f}%)", use_container_width=True)
                for box in results.boxes:
                    cls = int(box.cls[0])
                    class_name = custom_model.names[cls]
                    if class_name in detection_responses:
                        st.session_state.stats["total_objects"] += 1
                        st.session_state.stats["object_counts"][class_name] = st.session_state.stats["object_counts"].get(class_name, 0) + 1
                        st.session_state.stats["alerts_triggered"] += 1
                        st.warning(detection_responses[class_name])

    elif input_type == "Random Train Image":
        if "random_image" not in st.session_state:
            st.session_state.random_image = os.path.join(image_folder, random.choice(os.listdir(image_folder)))
        image_path = st.session_state.random_image
        st.image(image_path, caption="Random Train Image", use_container_width=True)
        col1, col2 = st.columns(2)
        if col1.button("🔄 Change Image"):
            st.session_state.random_image = os.path.join(image_folder, random.choice(os.listdir(image_folder)))
            st.rerun()
        if col2.button("🚀 Detect"):
            results = custom_model(image_path)[0]
            plotted = results.plot()
            confs = results.boxes.conf
            avg_conf = float(confs.mean()) * 100 if confs is not None else 0.0
            st.image(plotted, caption=f"🔍 Detection Result (Confidence: {avg_conf:.2f}%)", use_container_width=True)
            for box in results.boxes:
                cls = int(box.cls[0])
                class_name = custom_model.names[cls]
                if class_name in detection_responses:
                    st.session_state.stats["total_objects"] += 1
                    st.session_state.stats["object_counts"][class_name] = st.session_state.stats["object_counts"].get(class_name, 0) + 1
                    st.session_state.stats["alerts_triggered"] += 1
                    st.warning(detection_responses[class_name])

    elif input_type == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            cap = cv2.VideoCapture(video_path)
            st.session_state.stats["video_runs"] += 1
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = video_model(frame)[0]
                annotated_frame = results.plot()
                stframe.image(annotated_frame, channels="BGR", use_container_width=True)
                for box in results.boxes:
                    cls = int(box.cls[0])
                    class_name = video_model.names[cls]
                    if class_name in detection_responses:
                        st.session_state.stats["total_objects"] += 1
                        st.session_state.stats["object_counts"][class_name] = st.session_state.stats["object_counts"].get(class_name, 0) + 1
                        st.session_state.stats["alerts_triggered"] += 1
                        st.warning(detection_responses[class_name])
            cap.release()

    elif input_type == "Webcam":
        alert_placeholder = st.empty()
        debug_placeholder = st.empty()
        result_queue = queue.Queue()  # Queue to sync UI updates
        
        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.frame_count = 0
                self.result_queue = result_queue

            def recv(self, frame):
                self.frame_count += 1
                if self.frame_count % 10 != 0:
                    return frame
                
                try:
                    image = frame.to_ndarray(format="bgr24")
                    results = video_model(image)[0]
                    detected_objects = []
                    
                    for box in results.boxes:
                        cls = int(box.cls[0])
                        class_name = video_model.names[cls]
                        if class_name in detection_responses:
                            st.session_state.stats["total_objects"] += 1
                            st.session_state.stats["object_counts"][class_name] = st.session_state.stats["object_counts"].get(class_name, 0) + 1
                            st.session_state.stats["alerts_triggered"] += 1
                            detected_objects.append(detection_responses[class_name])
                    
                    if detected_objects:
                        self.result_queue.put(detected_objects[0])
                    
                    debug_placeholder.text(f"Processed frame {self.frame_count}: {len(detected_objects)} objects")
                    return av.VideoFrame.from_ndarray(results.plot(), format="bgr24")
                except Exception as e:
                    debug_placeholder.text(f"Error: {str(e)}")
                    return frame

        st.session_state.stats["webcam_usage"] += 1
        ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
        
        # Update UI from queue
        if ctx.video_processor:
            try:
                while True:
                    alert = result_queue.get_nowait()
                    if alert and st.session_state.stats["last_alert"] != alert:
                        st.session_state.stats["last_alert"] = alert
                        alert_placeholder.warning(alert)
            except queue.Empty:
                pass

    elif input_type == "SOS Alert":
        sos_alert_ui()

with tabs[2]:
    st.subheader("💬 Customer Support Chatbot")
    st.markdown("How Can We Help You With?")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    manual_responses = {
        "hi": "Hello! How can I assist you with road safety today?",
        "hello": "Hi there! Ask me anything about ADAS or road detection.",
        "help": "You can ask me about road safety, object detection, or how to use this app.",
        "what is adas": "ADAS stands for Advanced Driver Assistance System, which includes features like object detection, lane keeping, and collision warnings.",
        "sos": "To send an SOS alert, go to the SOS tab and press 'Trigger SOS'. Stay calm and wait for assistance.",
        "object detected": "We use a YOLOv8 model to detect objects like vehicles, humans, animals, and obstacles.",
        "how accurate": "Detection accuracy depends on lighting, angle, and object visibility. We aim to maintain over 90% with proper training.",
        "feedback": "We appreciate your feedback! Use the chat to share your experience or improvement suggestions.",
        "bye": "Drive safe! Hope to assist you again soon.",
        "thank you": "You're welcome! Happy to help.",
        "exit": "You can close the app or tab when done. Your data is safe.",
        "which model": "We're using a custom-trained YOLOv8 model tailored for road object detection.",
        "version": "You're using the latest beta version of the ADAS Assistant with YOLOv8 integration.",
        "who made this": "This app was developed as part of a major ADAS project using Streamlit and Ultralytics YOLOv8.",
        "can you detect animals": "Yes! Our model detects animals, humans, and other road objects in real-time.",
        "video not working": "Please ensure the uploaded video is in .mp4, .mov, or .avi format and not corrupted.",
        "webcam not working": "Make sure your camera is enabled and not used by another app. Reload the tab if needed.",
        "random image": "Go to the Image Detection tab and click 'Generate Random Image' to try it!",
        "detect": "Detection is handled by our YOLOv8 model. Upload an image, video, or use the webcam to see it in action!",
        "animal": "Yes, we can detect animals like dogs and cats on the road. Check the webcam or video tab!",
        "car": "Our system detects vehicles like cars and trucks to help you stay safe on the road.",
        "slow": "If the app is slow, try refreshing or reducing input size (e.g., smaller videos or fewer frames).",
        "hang": "If the webcam hangs, ensure your system has enough resources. Try closing other apps or using a lighter model.",
        "error": "If you see an error, note the message and let me know—I’ll help fix it!",
        "test": "To test, try uploading an image or video with clear objects like cars or people.",
        "support": "I’m here to support you! Ask about features, troubleshooting, or anything else."
    }

    user_input = st.text_input("You:", key="chat_input")

    if user_input:
        prompt = user_input.lower().strip()
        bot_response = None
        for key, value in manual_responses.items():
            if key in prompt or any(word in key for word in prompt.split()):
                bot_response = value
                break
        if not bot_response:
            bot_response = "I'm sorry, I didn't catch that. Could you please rephrase?"

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", bot_response))

    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")

st.markdown("""
    <div class='footer'>
        © 2025 ADAS Road Detection System
    </div>
""", unsafe_allow_html=True)
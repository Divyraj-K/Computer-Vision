import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Load emotion model
model = load_model("model/emotion_model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.set_page_config(page_title="Face Emotion Detection", layout="centered")
st.title("😊 Live Face Emotion Detection")
st.markdown("Detect human emotions in real-time using **Deep Learning + OpenCV**")

# Video Transformer
class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)

            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, emotion, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        return img

# Start webcam
webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False}
)

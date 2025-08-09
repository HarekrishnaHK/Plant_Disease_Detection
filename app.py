import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from datetime import datetime
from tensorflow.keras.models import load_model

# ----------------------------- CONFIGURATION -----------------------------
st.set_page_config(page_title="🌿 Plant Disease Detector", page_icon="🌱", layout="centered")

# Load the trained model
model = load_model("plant_disease_cnn.h5", compile=False)
class_names = ["Healthy", "Powdery", "Rust"]  # Update with your actual class labels

# ----------------------------- STYLING -----------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #fffde7);
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00695c;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 30px;
    }
    .feedback-box {
        background-color: #ffffffcc;
        padding: 20px;
        border-radius: 10px;
        margin-top: 30px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------- HEADER -----------------------------
st.markdown('<div class="title">🌿 Plant Disease Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload one or more plant leaf images to detect diseases</div>', unsafe_allow_html=True)
st.markdown("---")

# ----------------------------- IMAGE UPLOAD -----------------------------
uploaded_files = st.file_uploader("📤 Upload Leaf Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
results = []

# ----------------------------- PREDICTION -----------------------------
if uploaded_files:
    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(rgb_image, caption=f"🖼️ {file.name}", use_column_width=True)
        st.markdown("🔍 **Analyzing Image...**")

        img = cv2.resize(image, (128, 128)) / 255.0
        img = np.expand_dims(img, axis=0)

        # Prediction with confidence threshold
        prediction = model.predict(img)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = round(float(np.max(prediction)) * 100, 2)

        if confidence < 70:
            st.warning("⚠️ Not confident – image may not be a leaf or disease is unclear.")
        else:
            st.success(f"✅ **Prediction:** {predicted_class}")
            st.progress(int(confidence))
            st.markdown(f"📊 **Confidence:** {confidence:.2f}%")

        results.append({
            "Image Name": file.name,
            "Prediction": predicted_class if confidence >= 70 else "Unknown / Low Confidence",
            "Confidence (%)": confidence
        })


    # ----------------------------- CSV DOWNLOAD -----------------------------
    df = pd.DataFrame(results)
    csv = df.to_csv(index=False).encode()

    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name="plant_disease_predictions.csv",
        mime="text/csv"
    )

# ----------------------------- FEEDBACK SECTION -----------------------------
st.markdown("---")
st.markdown("<div class='feedback-box'><h4>💬 Feedback</h4>", unsafe_allow_html=True)
user_feedback = st.text_area("Let us know what you think about the app:")

if st.button("Submit Feedback"):
    if not user_feedback.strip():
        st.warning("⚠️ Please write something before submitting.")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("feedback.csv", "a") as f:
            f.write(f"{timestamp},{user_feedback.strip()}\n")
        st.success("✅ Thank you for your feedback!")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------- VIEW FEEDBACK -----------------------------
st.markdown("### 📂 View Previous Feedback")

if os.path.exists("feedback.csv"):
    feedback_data = pd.read_csv("feedback.csv", names=["Timestamp", "Feedback"])
    st.dataframe(feedback_data)
else:
    st.info("No feedback submitted yet.")

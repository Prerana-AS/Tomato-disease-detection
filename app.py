import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import os
from io import BytesIO
from PIL import Image

# ---------------------------
# Function to download model from Google Drive
# ---------------------------
def download_from_drive(drive_url, save_path):
    file_id = drive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    if not os.path.exists(save_path):
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            raise ValueError(f"Failed to download file: status_code={response.status_code}")

# ---------------------------
# Load Keras model
# ---------------------------
@st.cache_data(show_spinner=True)
def load_model(model_path, drive_url):
    download_from_drive(drive_url, model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# ---------------------------
# App layout
# ---------------------------
st.set_page_config(page_title="Tomato Disease Detection", page_icon="🍅", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: green;'>🍅 Tomato Disease Detection</h1>
    <p style='text-align: center;'>Upload a tomato leaf image and get predictions instantly.</p>
    """, unsafe_allow_html=True
)

# ---------------------------
# Upload image
# ---------------------------
uploaded_file = st.file_uploader("🔍 Choose an image...", type=["jpg", "jpeg", "png"])

# ---------------------------
# Load model
# ---------------------------
drive_link = "https://drive.google.com/file/d/1NA4PApABfAwAtq3rPZbVksOP79dH-en-/view?usp=drive_link"
model_path = "tmodel.keras"
model = load_model(model_path, drive_link)

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    img = img.resize((224, 224))  # adjust according to your model input
    x = image.img_to_ar

# Predict button with leaf icon
if st.button("🍃 Predict Disease", help="Click to detect the disease in the leaf"):
    class_idx, confidence = predict_disease(img)
    st.success(f"✅ Prediction: {classes[class_idx]}")
    st.info(f"📊 Confidence: {confidence*100:.2f}%")






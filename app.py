import os
import zipfile
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Tomato Disease Detection", layout="centered")
st.title("🍅 Tomato Leaf Disease Detection")

import gdown


ZIP_ID = "1n-XqG0ZDT_8BxiErB1zVfEDB9ZliXWdr"
ZIP_NAME = "model.zip"
MODEL_DIR = "saved_model"

# --- DOWNLOAD ZIP ---
if not os.path.exists(ZIP_NAME):
    st.info("📥 Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={ZIP_ID}", ZIP_NAME, quiet=False)
    st.success("✅ Zip downloaded!")

# --- EXTRACT ZIP ---
if not os.path.exists(MODEL_DIR):
    st.info("📂 Extracting model files...")
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(".")
    st.success("✅ Model extracted!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_DIR)
model = load_model()
st.success("🎉 Model loaded successfully!")


# --- Class names ---
CLASS_NAMES = [
    "Bacterial Spot", "Early Blight", "Late Blight",
    "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
    "Target Spot", "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus", "Healthy"
]

# --- User upload and prediction ---
uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img = img.resize((224,224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    with st.spinner("🔍 Predicting..."):
        pred = model.predict(img_array)
        idx = int(np.argmax(pred))
    st.success(f"🌿 Predicted Disease: **{CLASS_NAMES[idx]}**")
    st.info(f"Confidence: {pred[0][idx]*100:.2f}%")

st.caption("Developed by Prerana A S")



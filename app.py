import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from zipfile import ZipFile
import json

# Title
st.set_page_config(page_title="Tomato Disease Detection", page_icon="🍅", layout="centered")
st.title("🍅 Tomato Disease Detection App")

# Model download & load
MODEL_PATH = "final_tomato_model.h5"
FILE_ID = "12s86ZMXau2AuR_7MewumFPiATrqw55JV"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model...")
    response = requests.get(DOWNLOAD_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    st.success("✅ Model downloaded!")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# Load class names
try:
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
except Exception:
    class_names = [
        "Bacterial Spot", "Early Blight", "Late Blight",
        "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
        "Target Spot", "Tomato Yellow Leaf Curl Virus",
        "Tomato Mosaic Virus", "Healthy"
    ]

# Image uploader
uploaded_file = st.file_uploader("🔍 Upload a tomato leaf image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224,224))  # adjust as your model expects
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict 🍃"):
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        idx = int(np.argmax(preds))
        cls = class_names[idx]
        conf = float(np.max(preds) * 100)
        st.success(f"🌿 Predicted Disease: **{cls}**")
        st.info(f"🧠 Confidence: {conf:.2f}%")

        if "healthy" in cls.lower():
            st.balloons()
            st.write("🎉 The plant looks healthy!")
        else:
            st.warning("⚠️ The plant appears affected.")

st.markdown("---")
st.caption("Developed by Prerana A S")

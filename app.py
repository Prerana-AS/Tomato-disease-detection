import os
import zipfile
import requests
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Tomato Disease Detection", layout="centered")
st.title("🍅 Tomato Leaf Disease Detection")

# ---- Model download and extraction ----
MODEL_ZIP_URL = "https://drive.google.com/uc?export=download&id=1n-XqG0ZDT_8BxiErB1zVfEDB9ZliXWdr"
ZIP_NAME = "saved_model_format.zip"
EXTRACT_DIR = "saved_model_extracted"

if not os.path.exists(ZIP_NAME):
    with st.spinner("🔽 Downloading model ZIP..."):
        r = requests.get(MODEL_ZIP_URL, stream=True)
        r.raise_for_status()
        with open(ZIP_NAME, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    st.success("✅ Model ZIP downloaded.")

if not os.path.exists(EXTRACT_DIR):
    with st.spinner("📂 Extracting model..."):
        with zipfile.ZipFile(ZIP_NAME, "r") as z:
            z.extractall(EXTRACT_DIR)
    st.success("✅ Model extracted.")

@st.cache_resource
def load_model():
    # If the extracted folder contains a single subfolder, use that path
    children = [c for c in os.listdir(EXTRACT_DIR) if not c.startswith("__")]
    if len(children) == 1 and os.path.isdir(os.path.join(EXTRACT_DIR, children[0])):
        real_path = os.path.join(EXTRACT_DIR, children[0])
    else:
        real_path = EXTRACT_DIR
    model = tf.keras.models.load_model(real_path, compile=False)
    return model

model = load_model()

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


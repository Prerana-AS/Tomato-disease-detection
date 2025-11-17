import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

st.set_page_config(page_title="Tomato Disease Detection", layout="centered")

st.title("🍅 Tomato Leaf Disease Detection")

# Model download
MODEL_PATH = "final_tomato_model.keras"
DRIVE_ID = "1SJRj_QI0rzWSAykQ2N94LzeSwiNBRcSa"
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_ID}"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    r = requests.get(DRIVE_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    st.success("Model downloaded!")

# Load model
@st.cache_resource
def load_model():
    m = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return m

model = load_model()

# Class names
CLASS_NAMES = [
    "Bacterial Spot", "Early Blight", "Late Blight",
    "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
    "Target Spot", "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus", "Healthy"
]

# Upload and predict
uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img = img.resize((224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        pred = model.predict(img_array)
        idx = int(np.argmax(pred))
    st.success(f"Prediction: **{CLASS_NAMES[idx]}**")
    st.info(f"Confidence: {pred[0][idx]*100:.2f}%")

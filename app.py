import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# ---------------------------
# Function to download model if not present
# ---------------------------
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        st.info("Downloading model...")
        file_id = model_url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(download_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded!")

# ---------------------------
# Load Keras model
# ---------------------------
@st.cache_resource
def load_model(model_path, model_url=None):
    if model_url:
        download_model(model_url, model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="Tomato Disease Detector", layout="centered")
st.title("🍅 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader("🔍 Upload a leaf image", type=["jpg", "jpeg", "png"])

model_path = "tmodel.keras"
drive_link = "https://drive.google.com/file/d/1NA4PApABfAwAtq3rPZbVksOP79dH-en-/view?usp=drive_link"

model = load_model(model_path, drive_link)

# ---------------------------
# Prediction
# ---------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))  # resize according to your model input
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    pred = model.predict(image_array)
    class_idx = np.argmax(pred, axis=1)[0]

    # Map your classes here
    classes = ["Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot",
               "Spider Mites", "Target Spot", "Tomato Yellow Leaf Curl Virus", "Two-Spotted Spider Mite", "Healthy"]
    st.subheader(f"Prediction: {classes[class_idx]}")

    
    # Predict button with leaf icon
    if st.button("🍃 Predict Disease", help="Click to detect the disease in the leaf"):
        class_idx, confidence = predict_disease(img)
        st.success(f"✅ Prediction: {classes[class_idx]}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")


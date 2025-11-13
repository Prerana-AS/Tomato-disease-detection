import os
import requests
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ----------------------------
# Title & description
# ----------------------------
st.set_page_config(page_title="Tomato Disease Detection", page_icon="🍅")
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload an image of a tomato leaf, and the model will predict its disease.")

# ----------------------------
# Download model from Drive
# ----------------------------
@st.cache_data(show_spinner=True)
def download_model(drive_url, save_path):
    if os.path.exists(save_path):
        return save_path
    file_id = drive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    else:
        raise ValueError(f"Failed to download file: status_code={response.status_code}")

# ----------------------------
# Load H5 model
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_h5_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

# ----------------------------
# Prediction function
# ----------------------------
def predict_image(model, img: Image.Image):
    img = img.resize((224, 224))  # adjust size to your model's input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    return class_idx, preds[0][class_idx]

# ----------------------------
# Main
# ----------------------------
drive_link = "https://drive.google.com/file/d/1CYYtsKoyVo9FhNVhnejeQH2ad69Md4P5/view?usp=drive_link"
model_path = "tmodel.h5"

# Download & load model
model_file = download_model(drive_link, model_path)
model = load_h5_model(model_file)

# File uploader
uploaded_file = st.file_uploader("🔍 Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    class_idx, confidence = predict_image(model, image)
    
    # Map indices to your classes
    classes = [
        "Bacterial Spot", "Early Blight", "Late Blight",
        "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
        "Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Healthy"
    ]
    
    st.write(f"**Prediction:** {classes[class_idx]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import os
from io import BytesIO
from PIL import Image

# ---------------------------
# Google Drive download function
# ---------------------------
def download_from_drive(drive_link, save_path):
    """
    Downloads a file from Google Drive given a share link.
    """
    file_id = drive_link.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    if not os.path.exists(save_path):
        response = requests.get(download_url)
        with open(save_path, "wb") as f:
            f.write(response.content)
    return save_path

# ---------------------------
# Load model function
# ---------------------------
@st.cache_resource
def load_model(model_path, drive_link):
    """
    Loads the Keras model from local path or downloads from Google Drive if not present.
    """
    # Download if not exists
    download_from_drive(drive_link, model_path)
    # Load Keras model
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

st.title("🍅 Tomato Disease Detection App")
st.write("Upload an image of a tomato leaf and get the predicted disease.")

# Upload image
uploaded_file = st.file_uploader("🔍 Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

# Model setup
drive_link = "https://drive.google.com/file/d/1NA4PApABfAwAtq3rPZbVksOP79dH-en-/view?usp=drive_link"
model_path = "tmodel.keras"

model = load_model(model_path, drive_link)

# Prediction
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = img.resize((224, 224))  # adjust size according to your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # normalize if your model was trained this way
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Replace these with your actual class names
    class_names = [
        "Bacterial Spot", "Early Blight", "Late Blight", 
        "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
        "Target Spot", "Tomato Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
    ]
    
    st.write(f"**Prediction:** {class_names[predicted_class]}")


    
    # Predict button with leaf icon
    if st.button("🍃 Predict Disease", help="Click to detect the disease in the leaf"):
        class_idx, confidence = predict_disease(img)
        st.success(f"✅ Prediction: {classes[class_idx]}")
        st.info(f"📊 Confidence: {confidence*100:.2f}%")



import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# ---------------------------
# Sidebar UI
# ---------------------------
st.sidebar.title("🍅 Tomato Disease Detection")
st.sidebar.write("Upload a tomato leaf image to detect the disease.")

# Upload button with magnifying glass icon
uploaded_file = st.sidebar.file_uploader("🔍 Choose an image", type=["jpg", "jpeg", "png"], help="Upload a tomato leaf image")

# ---------------------------
# Load Keras model from Google Drive
# ---------------------------
@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?id=1NA4PApABfAwAtq3rPZbVksOP79dH-en-"
    response = requests.get(model_url)
    model_path = "tmodel.keras"
    with open(model_path, "wb") as f:
        f.write(response.content)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# ---------------------------
# Prediction logic
# ---------------------------
def predict_disease(img):
    img = img.resize((256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    return class_idx, confidence

# Mapping classes
classes = [
    "Bacterial Spot", 
    "Early Blight", 
    "Late Blight", 
    "Leaf Mold", 
    "Septoria Leaf Spot", 
    "Spider Mites", 
    "Target Spot", 
    "Tomato Yellow Leaf Curl Virus", 
    "Mosaic Virus", 
    "Healthy"
]

# ---------------------------
# Main app
# ---------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Predict button with leaf icon
    if st.sidebar.button("🍃 Predict Disease", help="Click to detect the disease in the leaf"):
        class_idx, confidence = predict_disease(img)
        st.sidebar.success(f"✅ Prediction: {classes[class_idx]}")
        st.sidebar.info(f"📊 Confidence: {confidence*100:.2f}%")

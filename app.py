import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# ---------------------------
# Title and description
# ---------------------------
st.set_page_config(page_title="Tomato Disease Detection", page_icon="🍅")
st.title("🍅 Tomato Disease Detection")
st.write("Upload a tomato leaf image and the model will predict its disease.")

# ---------------------------
# Load Keras model from Google Drive
# ---------------------------
def load_model(model_path="tmodel.keras", drive_link="https://drive.google.com/uc?id=1NA4PApABfAwAtq3rPZbVksOP79dH-en-"):
    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive...")
        gdown.download(drive_link, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# ---------------------------
# Image upload
# ---------------------------
uploaded_file = st.file_uploader("🔍 Upload Tomato Leaf Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # ---------------------------
    # Prediction
    # ---------------------------
    st.write("Predicting...")
    img = img.resize((224, 224))  # Resize according to model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    
    # Replace with your actual class labels
    class_labels = [
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
    
    st.success(f"Prediction: {class_labels[class_idx]} ({predictions[0][class_idx]*100:.2f}% confidence)")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Developed with ❤️ using TensorFlow & Streamlit")

# Predict button with leaf icon
if st.button("🍃 Predict Disease", help="Click to detect the disease in the leaf"):
    class_idx, confidence = predict_disease(img)
    st.success(f"✅ Prediction: {classes[class_idx]}")
    st.info(f"📊 Confidence: {confidence*100:.2f}%")





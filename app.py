import os
import requests
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import gdown

# Import DTypePolicy from Keras 3 (for compatibility)
try:
    from keras.dtype_policies import DTypePolicy
except ImportError:
    st.error("Keras 3 not installed. Run 'pip install --upgrade keras' and restart.")
    st.stop()

# 🏷️ Title
st.title("🍅 Tomato Disease Detection")

# 📁 Model path
MODEL_PATH = "tmodel.h5"

# Replace with your Google Drive file ID
FILE_ID = "1CYYtsKoyVo9FhNVhnejeQH2ad69Md4P5"
if not os.path.exists(MODEL_PATH):
    st.info("⏬ Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

# ✅ Custom InputLayer to fix old model config issues
from tensorflow.keras.layers import InputLayer

class CustomInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return super().from_config(config)

@st.cache_resource
def load_model_cached():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            'InputLayer': CustomInputLayer,
            'DTypePolicy': DTypePolicy
        }
    )
    return model

model = load_model_cached()

# 🗂️ Load class names
try:
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
except Exception as e:
    st.error(f"Error loading class names: {e}")
    class_names = [
        "Bacterial Spot", "Early Blight", "Late Blight",
        "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
        "Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Healthy"
    ]

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((256, 256))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        # Preprocess image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[class_index]
        confidence = np.max(prediction) * 100

        # Display result
        st.success(f"🌿 Predicted Disease: **{predicted_class}**")
        st.info(f"🧠 Confidence: {confidence:.2f}%")

        if "healthy" in predicted_class.lower():
            st.balloons()
            st.write("🎉 The plant looks healthy!")
        else:
            st.warning("⚠️ The plant seems affected. Consider checking treatment options.")

st.markdown("---")
st.caption("Developed by Prerana A S")


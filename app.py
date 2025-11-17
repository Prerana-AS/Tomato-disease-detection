import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import gdown

# Title
st.title("🍅 Tomato Disease Detection")

# Model Path
MODEL_PATH = "tmodel.h5"

# Google Drive file ID
FILE_ID = "1CYYtsKoyVo9FhNVhnejeQH2ad69Md4P5"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    st.info("⏬ Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    st.success("✅ Model downloaded successfully!")

@st.cache_resource
def load_model_cached():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        safe_mode=False,     # IMPORTANT FIX
        compile=False
    )
    return model

model = load_model_cached()

# Load class names
try:
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())   # FIXED
except:
    class_names = [
        "Bacterial Spot", "Early Blight", "Late Blight",
        "Leaf Mold", "Septoria Leaf Spot", "Spider Mites",
        "Target Spot", "Tomato Yellow Leaf Curl Virus",
        "Tomato Mosaic Virus", "Healthy"
    ]

# File uploader
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict Disease"):
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        predicted_class = class_names[class_index]
        confidence = np.max(prediction) * 100

        st.success(f"🌿 Predicted Disease: **{predicted_class}**")
        st.info(f"🧠 Confidence: {confidence:.2f}%")

        if "healthy" in predicted_class.lower():
            st.balloons()
            st.write("🎉 The plant looks healthy!")
        else:
            st.warning("⚠️ The plant appears affected.")

st.markdown("---")
st.caption("Developed by Prerana A S")

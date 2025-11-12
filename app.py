import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import h5py

# ──────────────────────────────
# 🌿 Streamlit page configuration
# ──────────────────────────────
st.set_page_config(page_title="🍅 Tomato Disease Detection", layout="centered")
st.title("🍅 Tomato Leaf Disease Detection using Deep Learning")
st.write("Upload a tomato leaf image to identify its disease.")

# ──────────────────────────────
# 📥 Model loader (with fallback)
# ──────────────────────────────
@st.cache_resource
def load_model():
    model_path = "tomato_model.h5"

    # Download model from Google Drive if not present
    if not os.path.exists(model_path):
        with st.spinner("🔄 Downloading model from Google Drive..."):
            url = "https://drive.google.com/uc?id=1CYYtsKoyVo9FhNVhnejeQH2ad69Md4P5"
            gdown.download(url, model_path, quiet=False)

    # Try to load with new Keras 3 API
    try:
        model = tf.keras.models.load_model(model_path, safe_mode=False)
    except Exception as e:
        st.warning("⚠️ Detected legacy Keras model format — attempting auto-conversion…")
        try:
            from keras.saving import legacy_h5_format
            f = h5py.File(model_path, "r")
            model = legacy_h5_format.load_model_from_hdf5(f, compile=False)
            f.close()

            # Save converted version so it loads faster next time
            converted_path = "tomato_model_converted.keras"
            model.save(converted_path, save_format="keras")
            st.success("✅ Model converted successfully!")
        except Exception as ex:
            st.error(f"❌ Model conversion failed: {ex}")
            raise ex

    return model


# Load the model (cached)
model = load_model()

# ──────────────────────────────
# 🌱 Class labels
# ──────────────────────────────
class_names = [
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
    "Healthy Tomato Leaf"
]

# ──────────────────────────────
# 🖼️ Image upload + prediction
# ──────────────────────────────
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Display results
    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
    st.markdown("---")

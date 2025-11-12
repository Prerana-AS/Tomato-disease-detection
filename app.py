import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
from keras import __version__ as keras_version

st.set_page_config(page_title="🍅 Tomato Disease Detection", layout="centered")
st.title("🍅 Tomato Leaf Disease Detection using Deep Learning")
st.write("Upload a tomato leaf image to identify its disease.")

# ✅ Cached model loader
@st.cache_resource
def load_model():
    model_path = "tomatod_model.h5"
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    except Exception as e:
        # Legacy fallback for older Keras models
        from keras.src.legacy.saving import legacy_h5_format
        model = legacy_h5_format.load_model_from_hdf5(model_path, compile=False)
    if not os.path.exists(model_path):
        with st.spinner("🔄 Downloading model from Google Drive..."):
            url = "https://drive.google.com/uc?id=1CYYtsKoyVo9FhNVhnejeQH2ad69Md4P5"
            gdown.download(url, model_path, quiet=False)
    # safe_mode=False allows loading older .h5 models on new TF versions
    model = tf.keras.models.load_model(model_path, safe_mode=False)
    return model

model = load_model()

# ✅ Class labels
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

# ✅ Image upload
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # ✅ Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.write(f"**Predicted Class:** {class_names[predicted_class]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
    st.markdown("---")




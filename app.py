import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import os
import gdown



# 🏷️ Title
st.title("🍅 Tomato Disease Detection")

# 📁 Model path
MODEL_PATH = "tomatod_model.h5"

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
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model_cached()


# 🗂️ Load class names
try:
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    class_names = list(class_indices.keys())
except Exception as e:
    st.error(f"Error loading class names: {e}")
    class_names = [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites_Two_spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]

# 📤 File uploader
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
identification={
    "Tomato___Bacterial_spot": 
    "• Small, dark, water-soaked spots appear on leaves, which later turn brown with a yellow halo. Leaves look rough, spotted, and may tear. Fruits show raised black scabby spots.",

    "Tomato___Early_blight": 
    "•Early blight can be identified by brown circular spots with clear concentric rings on tomato leaves, usually starting from the lower leaves and spreading upward. The area around the spots turns yellow, and the leaves eventually dry and fall off, making these target-like lesions the key sign to confirm the disease.",

    "Tomato___Late_blight":
    "• Late blight shows *large, dark, water-soaked patches* on leaves that spread very fast, often with a *white fuzzy growth* on the underside in humid weather. Leaves collapse quickly, and stems or fruits may also turn brown and rot, making the disease easy to recognize by its rapid damage and irregular wet-looking spots.",

    "Tomato___Leaf_Mold":
    "•Yellow patches form on the upper leaf surface, and the underside develops *olive-green or gray velvety mold*. Mostly affects older leaves in humid conditions.",

    "Tomato___Septoria_leaf_spot":
    "•Septoria shows *many small, round brown spots* with *light gray centers* and dark borders, mostly on *lower leaves. Spots are tiny (pinhead size), numerous, and do **not have concentric rings*, which helps differentiate it from early blight.",

    "Tomato___Spider_mites_Two_spotted_spider_mite":
    "•Leaves show tiny yellow speckles (stippling), eventually turning bronze or dry. Fine webbing is visible on the underside of leaves, especially in hot, dry conditions.",

    "Tomato___Target_Spot":
    "• Large brown spots with *distinct concentric rings*, similar to early blight but usually larger and more irregular. Spots expand rapidly in warm, wet conditions.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
    "•This virus causes *upward curling yellow leaves*, stunted growth, and very few flowers or fruits. Leaves become small, brittle, and bunch at the top like a rosette.",

    "Tomato___Tomato_mosaic_virus":
    "•Mosaic virus causes *mottled light and dark green patterns, distorted or narrow leaves, and sometimes **shriveled, hard fruits*. Growth becomes stunted, and patterns look like “mosaic tiles.",

    "Tomato___healthy":
    "• Plant is healthy!"
}
    
# 🌿 Remedies for each tomato leaf disease
remedies = {
    "Tomato___Bacterial_spot": 
    "•Remove infected leaves, avoid overhead watering, and improve airflow. Spray copper-based bactericides. Use disease-free seeds, rotate crops, and avoid touching wet plants.",

    "Tomato___Early_blight": 
    "•To control it, remove infected leaves, avoid wetting the foliage, and ensure good airflow. Spray *neem oil, copper fungicide, or mancozeb* to stop the spread. Use mulch to prevent soil splash, water only at the base, and maintain proper spacing to keep the plants healthy and protected.",

    "Tomato___Late_blight":
    "•To manage it, remove heavily infected leaves, avoid overhead watering, and improve ventilation. Spray *copper fungicide or mancozeb* for control, keep foliage dry, and avoid touching wet plants. For prevention, use resistant varieties, space plants well, and avoid planting tomatoes in the same soil every year.",

    "Tomato___Leaf_Mold":
    "•Increase ventilation, reduce humidity, and water at the base. Use fungicides like copper or chlorothalonil. Space plants properly and avoid overcrowding.",

    "Tomato___Septoria_leaf_spot":
    "•Remove infected leaves, avoid overhead watering, improve airflow, and use fungicides like *copper* or *mancozeb*. Mulch the soil, keep plants spaced, and avoid touching plants when wet.",

    "Tomato___Spider_mites_Two_spotted_spider_mite":
    "•Spray neem oil or insecticidal soap; wash leaves with strong water pressure. Increase humidity, remove heavily damaged leaves, and keep plants well-watered.",

    "Tomato___Target_Spot":
    "•Remove infected leaves and improve airflow. Use fungicides like mancozeb or chlorothalonil. Reduce leaf wetness, mulch the soil, and avoid overcrowding.",

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":
    "•Remove infected plants, control *whiteflies* (main carrier) using neem oil or sticky traps, cover plants with insect net, and avoid planting tomatoes near heavily infested areas. Use resistant varieties whenever possible.",

    "Tomato___Tomato_mosaic_virus":
    "•There is *no cure*; remove infected plants immediately. Disinfect tools, wash hands after handling tobacco products, control pests, and grow resistant varieties.",

    "Tomato___healthy":
    "• Continue regular watering, sunlight and nutrients."
}


if uploaded_file is not None:
    img = Image.open(uploaded_file)
    # 🔥 FIX: Convert RGBA or grayscale to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((128, 128))

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
        st.write("🌱 Identification:")
        st.info(identification[predicted_class])
         
        if "healthy" in predicted_class.lower():
            st.balloons()
            st.write("🎉 The plant looks healthy!")
        else:
            st.warning("⚠️ The plant is affected. Consider checking treatment options.")
            # Show remedy
        st.write("🌱 Recommended Remedy:")
        st.info(remedies[predicted_class])
        

        st.title("Tomato Leaf Disease Classification – Model Evaluation")

# Accuracy Curve
        st.subheader("Accuracy and loss Curve")
        st.image("accuracy and loss.jpg")



# Confusion Matrix
        st.subheader("Confusion Matrix")
        st.image("confusion matrix.jpg")



st.markdown("---")
st.caption("Developed by Prerana A S")









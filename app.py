import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
import os

# Define class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Download model from Google Drive if not present
MODEL_PATH = "modelenv.v1.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1KkDKRpFtvqiyPcx9yahqWy5lVBhBQ1Ct"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Streamlit UI
st.title("🌍 Environment Image Classifier")
st.markdown("Upload a satellite or landscape image and I'll tell you if it's **Cloudy**, **Desert**, **Green Area**, or **Water**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_batch)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = 100 * np.max(prediction)

    st.subheader("Prediction")
    st.success(f"**{predicted_class}** with {confidence:.2f}% confidence.")

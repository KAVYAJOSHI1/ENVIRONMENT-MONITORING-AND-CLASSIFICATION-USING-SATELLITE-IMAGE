import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# Set title and description
st.set_page_config(page_title="Environment Classifier", layout="centered")
st.title("🌍 Environment Image Classifier")
st.markdown("Upload a satellite/landscape image and get a prediction of whether it's **Cloudy**, **Desert**, **Green Area**, or **Water**.")

# Class labels
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Model file info
MODEL_PATH = "modelenv.v1.h5"
FILE_ID = "1KkDKRpFtvqiyPcx9yahqWy5lVBhBQ1Ct"  # Google Drive file ID

# Function to download and load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# Load model once
model = load_model()

# Upload image
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize and normalize image
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_batch)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index]) * 100

    # Show prediction
    st.subheader("📊 Prediction")
    st.success(f"**{predicted_class}** with **{confidence:.2f}%** confidence.")

    # Optional: Show probabilities
    st.subheader("🔍 Class Probabilities")
    for i, score in enumerate(predictions):
        st.write(f"{class_names[i]}: {score * 100:.2f}%")

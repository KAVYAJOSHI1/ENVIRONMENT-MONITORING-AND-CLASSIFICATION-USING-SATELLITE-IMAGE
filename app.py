import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration with custom styling
st.set_page_config(
    page_title="Environment Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🌍"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    .main {
        padding: 2rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        opacity: 0.9;
        line-height: 1.6;
    }
    
    /* Upload section */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 2px dashed #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    /* Results section */
    .results-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .confidence-score {
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .class-name {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Image display */
    .image-container {
        text-align: center;
        margin: 2rem 0;
    }
    
    .uploaded-image {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        max-width: 100%;
        height: auto;
    }
    
    /* Probability bars */
    .prob-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    
    .prob-bar {
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 10px;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide streamlit components */
    .css-1d391kg {
        display: none;
    }
    
    .css-1v0mbdj {
        display: none;
    }
    
    /* Custom file uploader */
    .stFileUploader > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        border: none;
    }
    
    .stFileUploader label {
        color: white !important;
        font-weight: 600;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header-container">
    <div class="header-title">🌍 Environment AI Classifier</div>
    <div class="header-subtitle">
        Harness the power of AI to classify satellite and landscape images<br>
        Detect <strong>Cloudy</strong>, <strong>Desert</strong>, <strong>Green Areas</strong>, and <strong>Water</strong> environments
    </div>
</div>
""", unsafe_allow_html=True)

# Class labels and emojis
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
class_emojis = {'Cloudy': '☁️', 'Desert': '🏜️', 'Green_Area': '🌲', 'Water': '🌊'}
class_colors = {'Cloudy': '#87CEEB', 'Desert': '#DEB887', 'Green_Area': '#228B22', 'Water': '#4682B4'}

# Model file info
MODEL_PATH = "modelenv.v1.h5"
FILE_ID = "1KkDKRpFtvqiyPcx9yahqWy5lVBhBQ1Ct"

# Function to download and load model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading AI model... This may take a moment."):
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# Load model
try:
    model = load_model()
    st.success("✅ AI Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Create columns for layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
    <div class="upload-section">
        <h3 style="text-align: center; color: #667eea; margin-bottom: 1rem;">📸 Upload Your Image</h3>
        <p style="text-align: center; color: #666; margin-bottom: 2rem;">
            Choose a satellite or landscape image to classify
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        help="Upload satellite or landscape images in JPG, JPEG, or PNG format"
    )

with col2:
    if uploaded_file:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process image
        with st.spinner("🔍 Analyzing image with AI..."):
            img_resized = image.resize((256, 256))
            img_array = np.array(img_resized) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_batch)[0]
            predicted_index = np.argmax(predictions)
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100
        
        # Results section
        st.markdown(f"""
        <div class="results-container">
            <div class="prediction-card">
                <div style="text-align: center;">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">
                        {class_emojis[predicted_class]}
                    </div>
                    <div class="class-name">{predicted_class.replace('_', ' ')}</div>
                    <div class="confidence-score">{confidence:.1f}%</div>
                    <div style="font-size: 1.1rem; opacity: 0.9;">Confidence Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Full-width probability visualization
if uploaded_file:
    st.markdown("## 📊 Detailed Analysis")
    
    # Create interactive probability chart
    fig = go.Figure()
    
    colors = [class_colors[class_name] for class_name in class_names]
    probabilities = [float(p) * 100 for p in predictions]
    
    fig.add_trace(go.Bar(
        x=[name.replace('_', ' ') for name in class_names],
        y=probabilities,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probabilities],
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Classification Probabilities',
            'x': 0.5,
            'font': {'size': 24, 'family': 'Poppins'}
        },
        xaxis_title="Environment Types",
        yaxis_title="Probability (%)",
        template="plotly_white",
        height=400,
        font=dict(family="Poppins", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100]),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    st.markdown("### 🔍 Classification Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 Prediction Accuracy",
            value=f"{confidence:.1f}%",
            delta=f"{confidence - 50:.1f}% above baseline"
        )
    
    with col2:
        second_highest = sorted(predictions, reverse=True)[1] * 100
        st.metric(
            label="📈 Second Best Match",
            value=f"{second_highest:.1f}%",
            delta=f"{confidence - second_highest:.1f}% difference"
        )
    
    with col3:
        certainty = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
        st.metric(
            label="✅ Certainty Level",
            value=certainty,
            delta=f"Based on {confidence:.1f}% confidence"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Powered by TensorFlow & Streamlit | 🌍 Environment Classification AI</p>
    <p>Upload satellite images to discover the environment type with advanced deep learning</p>
</div>
""", unsafe_allow_html=True)

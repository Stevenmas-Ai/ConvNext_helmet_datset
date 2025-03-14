import streamlit as st
import requests
from PIL import Image
import io

# Set FastAPI URL
API_URL = "http://backend:8000/predict/"

# Streamlit UI
st.title("Helmet Detection 🚧")
st.write("Upload an image to check if a person is wearing a helmet.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Send image to FastAPI for prediction
    with st.spinner("Analyzing..."):
        response = requests.post(API_URL, files={"file": img_bytes})
        result = response.json()

    # Display prediction
    st.success(f"Prediction: **{result['prediction']}**")
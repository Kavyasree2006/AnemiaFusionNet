
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from models.image_model import ImageModel

st.title("🔍 Predict Anemia")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png"])

age = st.number_input("Age", 1, 100, 25)
bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

gender = st.selectbox("Gender", ["Male", "Female"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])
diet = st.selectbox("Diet", ["Vegetarian", "Mixed"])

geo_risk = st.slider("Geo Risk Score", 0.0, 1.0, 0.5)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Eye Image", width=300)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)

    device = torch.device("cpu")

    model = ImageModel()

    model.load_state_dict(
        torch.load("outputs/image_model.pth", map_location=device)
    )

    model.eval()

    with torch.no_grad():

        output = model(image_tensor)

        prob = torch.sigmoid(output).item()

    st.subheader("Prediction Result")

    if prob > 0.5:
        st.error(f"⚠️ Anemic Detected ({prob:.2f})")
    else:
        st.success(f"✅ Non-Anemic ({prob:.2f})")

    st.progress(min(max(prob, 0.0), 1.0))

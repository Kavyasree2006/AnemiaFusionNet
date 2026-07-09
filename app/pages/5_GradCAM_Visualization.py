
import streamlit as st
import torch
from PIL import Image

from models.image_model import ImageModel
from app.utils.gradcam import generate_gradcam

st.title("🔥 Grad-CAM Visualization")

uploaded_file = st.file_uploader(
    "Upload Eye Image",
    type=["jpg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image)

    model = ImageModel()

    model.load_state_dict(
        torch.load(
            "outputs/image_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    cam = generate_gradcam(
        model,
        image
    )

    st.image(
        cam,
        caption="Model Attention Map"
    )

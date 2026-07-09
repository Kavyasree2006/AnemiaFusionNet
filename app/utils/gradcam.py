import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(model, image):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    input_tensor = transform(image).unsqueeze(0)

    target_layer = model.backbone.features[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    rgb_img = np.array(
        image.resize((224, 224))
    ) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.fusion_dataset import FusionDataset
from preprocessing.transforms import train_transform, val_transform
from models.fusion_model import FusionModel

# DATASETS
train_dataset = FusionDataset(
    "dataset/train.csv",
    "dataset/images",
    train_transform
)

val_dataset = FusionDataset(
    "dataset/val.csv",
    "dataset/images",
    val_transform
)

# DATALOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16
)

# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# MODEL
model = FusionModel(
    clinical_input_dim=7
).to(device)

# LOSS
criterion = nn.BCEWithLogitsLoss()

# OPTIMIZER
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 10

# TRAINING LOOP
for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, clinical, geo, labels in train_loader:

        images = images.to(device)

        clinical = clinical.to(device)

        geo = geo.to(device)

        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(
            images,
            clinical,
            geo
        )

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# SAVE MODEL
torch.save(
    model.state_dict(),
    "outputs/fusion_model.pth"
)

print("\nFusion model saved successfully")
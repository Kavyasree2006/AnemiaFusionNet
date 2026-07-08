import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing.dataset_loader import AnemiaDataset
from preprocessing.transforms import val_transform
from models.image_model import ImageModel

# DATASET
test_dataset = AnemiaDataset(
    "dataset/test.csv",
    "dataset/images",
    val_transform
)

test_loader = DataLoader(test_dataset, batch_size=16)

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL
model = ImageModel().to(device)

model.load_state_dict(torch.load("outputs/image_model.pth"))

model.eval()

all_labels = []
all_preds = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        preds = torch.sigmoid(outputs)

        preds = (preds > 0.5).int()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())

# METRICS
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

cm = confusion_matrix(all_labels, all_preds)

print("\n===== IMAGE MODEL RESULTS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

with open("outputs/image_results.txt", "w") as f:

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")

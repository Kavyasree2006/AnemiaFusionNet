import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from preprocessing.fusion_dataset import FusionDataset
from preprocessing.transforms import val_transform
from models.fusion_model import FusionModel

# DATASET
test_dataset = FusionDataset(
    "dataset/test.csv",
    "dataset/images",
    val_transform
)

test_loader = DataLoader(
    test_dataset,
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

model.load_state_dict(
    torch.load("outputs/fusion_model.pth")
)

model.eval()

all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():

    for images, clinical, geo, labels in test_loader:

        images = images.to(device)

        clinical = clinical.to(device)

        geo = geo.to(device)

        outputs = model(
            images,
            clinical,
            geo
        )

        probs = torch.sigmoid(outputs)

        preds = (probs > 0.5).int()

        all_labels.extend(labels.numpy())

        all_preds.extend(preds.cpu().numpy())

        all_probs.extend(probs.cpu().numpy())

# METRICS
accuracy = accuracy_score(all_labels, all_preds)

precision = precision_score(all_labels, all_preds)

recall = recall_score(all_labels, all_preds)

f1 = f1_score(all_labels, all_preds)

roc_auc = roc_auc_score(all_labels, all_probs)

cm = confusion_matrix(all_labels, all_preds)

print("\n===== FUSION MODEL RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")

print(cm)

# SAVE RESULTS
with open("outputs/fusion_results.txt", "w") as f:

    f.write(f"Accuracy : {accuracy:.4f}\n")

    f.write(f"Precision: {precision:.4f}\n")

    f.write(f"Recall   : {recall:.4f}\n")

    f.write(f"F1 Score : {f1:.4f}\n")

    f.write(f"ROC-AUC  : {roc_auc:.4f}\n")

print("\nResults saved successfully")
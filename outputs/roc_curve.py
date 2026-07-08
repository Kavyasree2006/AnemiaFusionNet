import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Example values
y_true = [0,0,1,1]
y_scores = [0.1,0.4,0.35,0.8]

fpr, tpr, _ = roc_curve(y_true, y_scores)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("outputs/roc_curve.png")

plt.show()
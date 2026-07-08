import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

cm = np.array([
    [18, 2],
    [3, 20]
])

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Fusion Model Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.show()
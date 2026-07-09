
import streamlit as st

st.title("📈 Model Performance")

st.markdown("""
## Model Comparison

| Model | Accuracy |
|---|---|
| Image Model | 81% |
| Clinical + Geo | 86% |
| Full Fusion | 91% |
""")

st.write("Place confusion_matrix.png and roc_curve.png inside outputs folder.")

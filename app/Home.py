
import streamlit as st

st.set_page_config(page_title="AnemiaFusionNet", layout="wide")

st.title("🩺 AnemiaFusionNet")
st.subheader("Multimodal AI-Based Region Aware Anemia Detection System")

st.markdown("""
## Features
- Eye Image Based Anemia Detection
- Clinical Data Analysis
- Geo-Risk Assessment
- Multimodal Feature Fusion
- Explainable AI Visualization
- Analytics Dashboard
- PDF Patient Reports
""")

col1, col2, col3 = st.columns(3)

col1.metric("Dataset Size", "218 Images")
col2.metric("Model Type", "Multimodal Fusion")
col3.metric("Prediction", "Anemia / Non-Anemia")

st.info("Use the sidebar to navigate through pages.")


import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("📊 Analytics Dashboard")

try:

csv_path = "dataset/full_dataset.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    fig1 = px.histogram(df, x="age")
    st.plotly_chart(fig1)

else:
    st.info(
        "Dataset is not included in the deployed app. "
        "Analytics dashboard is available locally."
    )

    fig1 = px.histogram(df, x="age", title="Age Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    if "hb" in df.columns:
        fig2 = px.histogram(df, x="hb", title="Hemoglobin Distribution")
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"Dataset loading error: {e}")

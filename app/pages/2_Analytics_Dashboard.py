
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Analytics Dashboard")

try:
    df = pd.read_csv("dataset/full_dataset.csv")

    fig1 = px.histogram(df, x="age", title="Age Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    if "hb" in df.columns:
        fig2 = px.histogram(df, x="hb", title="Hemoglobin Distribution")
        st.plotly_chart(fig2, use_container_width=True)

except Exception as e:
    st.error(f"Dataset loading error: {e}")

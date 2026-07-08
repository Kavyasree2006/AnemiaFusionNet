
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("🗺️ Geo Risk Analysis")

try:

csv_path = "dataset/full_dataset.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    # plots here

else:
    st.info(
        "Geo-risk analytics require the local dataset."
    )

    if "state" in df.columns and "geo_risk" in df.columns:

        fig = px.bar(
            df,
            x="state",
            y="geo_risk",
            title="State-wise Geo Risk"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("State or geo_risk column missing.")

except Exception as e:
    st.error(e)

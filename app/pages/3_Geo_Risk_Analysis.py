
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🗺️ Geo Risk Analysis")

try:
    df = pd.read_csv("dataset/full_dataset.csv")

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

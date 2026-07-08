
import streamlit as st
from fpdf import FPDF

st.title("📄 Patient Report Generator")

patient_name = st.text_input("Patient Name")

prediction = st.selectbox(
    "Prediction",
    ["Anemic", "Non-Anemic"]
)

if st.button("Generate PDF Report"):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, txt="AnemiaFusionNet Patient Report", ln=True)

    pdf.cell(200, 10, txt=f"Patient: {patient_name}", ln=True)

    pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)

    output_path = "patient_report.pdf"

    pdf.output(output_path)

    st.success("PDF Report Generated")

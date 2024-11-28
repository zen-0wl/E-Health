import re
import streamlit as st
import pdfplumber
import pandas as pd
import pickle
import base64

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to parse extracted text based on provided format
def parse_extracted_text(text):
    patterns = {
        #"Gender (0-M;1-F)": r"Gender:\s*(0|1)",  # Gender (0-M;1-F)
        "Blood Pressure (systolic)": r"Blood Pressure \(systolic\):\s*(\d+)",
        "Blood Pressure (diastolic)": r"Blood Pressure \(diastolic\):\s*(\d+)",
        "Heart Rate (bpm)": r"Heart Rate \(bpm\):\s*(\d+)",
        "Hemoglobin A1c (%)": r"Hemoglobin A1c \(%\):\s*(\d+\.?\d*)",
        "Breathing Rate (brpm)": r"Breathing Rate \(brpm\):\s*(\d+)",
        "Oxygen Saturation (%)": r"Oxygen Saturation \(%\):\s*(\d+\.?\d*)",
        "HRV SDNN (ms)": r"HRV SDNN \(ms\):\s*(\d+\.?\d*)",
        "RMSSD (ms)": r"RMSSD \(ms\):\s*(\d+\.?\d*)",
        "Recovery Ability": r"Recovery Ability:\s*(\d+)",
        "Stress Index": r"Stress Index:\s*(\d+\.?\d*)",
        "SNS Index": r"SNS Index:\s*(\d+\.?\d*)",
        "PNS Index": r"PNS Index:\s*(\d+\.?\d*)",
        "Hemoglobin (g/dl)": r"Hemoglobin \(g/dl\):\s*(\d+\.?\d*)",
        "Mean RRI (ms)": r"Mean RRI \(ms\):\s*(\d+\.?\d*)",
        "SD1 (ms)": r"SD1 \(ms\):\s*(\d+\.?\d*)",
        "SD2 (ms)": r"SD2 \(ms\):\s*(\d+\.?\d*)",
    }

    parsed_data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)  # Support multi-line patterns
        if match:
            parsed_data[key] = float(match.group(1))
        else:
            parsed_data[key] = None  # Set to None if not found

    return parsed_data

# Function to prepare input data for the model
def prepare_input(parsed_data):
    input_data = pd.DataFrame([parsed_data])
    return input_data

def show_pdf(file):
    pdf_file = file.getvalue()  # Get the file content
    pdf_base64 = base64.b64encode(pdf_file).decode('utf-8')  # Encode as base64
    pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="500" height="500"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit app setup
st.set_page_config(page_title="Healthcare Disease Prediction", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>HealthCare Binah.ai Disease Prediction</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Drag & Drop PDF here", type=["pdf"])

if uploaded_file is not None:
    # Display the uploaded file name
    st.sidebar.write(f"Uploaded file: {uploaded_file.name}")

    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(uploaded_file)

    # Layout: Original PDF and Extracted Text
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original Source Data:")
        show_pdf(uploaded_file)

    with col2:
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.subheader("Extracted Text:")
        st.text_area("Text Extraction", extracted_text, height=300)

    # Parsing and model prediction
    parsed_data = parse_extracted_text(extracted_text)
    st.subheader("Parsed Data:")
    st.json(parsed_data)

    input_data = prepare_input(parsed_data)
    
    # Ensure the model is loaded (replace 'best_model.pkl' with your actual model file)
    with open("random_forest_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    prediction = best_model.predict(input_data)

    # Display prediction result
    st.subheader("Model Prediction Result:")
    st.write(f"**Predicted Disease:** {prediction[0]}")
else:
    st.sidebar.info("Please upload a PDF file to get started.")
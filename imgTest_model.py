import re
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import pickle
import base64

patterns = {
        "Gender (0-M;1-F)": r"Gender.*?:\s*(0|1)",
        "Blood Pressure (systolic)": r"Blood Pressure\s+(\d+\.\d+|\d+)\s*/",
        "Blood Pressure (diastolic)": r"Blood Pressure\s+\d+\.\d+|\d+\s*/(\d+\.\d+|\d+)",
        "Heart Rate (bpm)": r"Heart Rate\s*[:\-]?\s*(\d+\.?\d*)\s*(bpm|bpm)?", 
        "Hemoglobin A1c (%)": r"Hemoglobin A1c\s*[:\-]?\s*(\d+\.?\d*)\s*(%)?",
        "Breathing Rate (brpm)": r"Breathing Rate\s*[:\-]?\s*(\d+\.?\d*)\s*(brpm|brpm)?",
        "Oxygen Saturation (%)": r"Oxygen Saturation\s*[:\-]?\s*(\d+\.?\d*)\s*(%)?",
        "HRV SDNN (ms)": r"HRV SDNN\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?",
        "RMSSD (ms)": r"RMSSD\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?",
        "Recovery Ability": r"Recovery Ability\s*[:\-]?\s*(\d+)",
        "Mean RRI (ms)": r"Mean RRI\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?",
        "Hemoglobin (g/dl)": r"Hemoglobin\s*[:\-]?\s*(\d+\.?\d*)\s*(g/dl)?",
        "Stress Index": r"Stress Index\s*[:\-]?\s*(\d+\.?\d*)",
        "SNS Index": r"SNS Index\s*[:\-]?\s*(-?\d+\.?\d*)",
        "PNS Index": r"PNS Index\s*[:\-]?\s*(-?\d+\.?\d*)",
        "SD1 (ms)": r"SD1\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?",
        "SD2 (ms)": r"SD2\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?"
    }

def extract_text_from_pdf(file):
    text = ""
    
    # First, try to extract text normally using pdfplumber
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                # If text extraction fails, use OCR on the image
                images = convert_from_bytes(file.getvalue())
                for img in images:
                    text += pytesseract.image_to_string(img)
    
    return text

# Function to parse extracted text based on provided format
def parse_extracted_text(text):
    parsed_data = {}
    for feature, pattern in patterns.items():
        match = re.search(pattern, text)  
        if match:
            parsed_data[feature] = float(match.group(1))
        else:
            parsed_data[feature] = None  
 
    return parsed_data


def prepare_input(parsed_data):
    
    trained_columns = [
        'Heart Rate (bpm)', 'Breathing Rate (brpm)', 'Oxygen Saturation (%)',
        'Blood Pressure (systolic)', 'Blood Pressure (diastolic)', 'Stress Index',
        'Recovery Ability', 'PNS Index', 'SNS Index', 'RMSSD (ms)', 'SD2 (ms)',
        'Hemoglobin A1c (%)', 'Mean RRI (ms)', 'SD1 (ms)', 'HRV SDNN (ms)',
        'Hemoglobin (g/dl)', 'Gender (0-M;1-F)'
    ]
    
    input_data = pd.DataFrame([parsed_data])
    
    input_data = input_data[trained_columns]
    
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

disease_labels = [
    "Anaemia",
    "Arrhythmias",
    "Atherosclerosis",
    "Autonomic Dysfunction",
    "Cardiovascular Disease (CVD)",
    "Chronic Fatigue Syndrome (CFS)",
    "Diabetes",
    "Healthy",
    "Hypertension",
    "Respiratory Disease (COPD or Asthma)",
    "Stress-related Disorders"
]

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

    with open("random_forest_model.pkl", "rb") as f:
        best_model = pickle.load(f)
        
    if input_data.isnull().all().all():
        st.write("**Prediction cannot be made due to missing data.**")
    else:
        prediction = best_model.predict(input_data)
        predicted_disease = disease_labels[prediction[0]]
        st.subheader("Model Prediction Result:")
        st.write(f"**Predicted Disease:** {predicted_disease}")
else:
    st.sidebar.info("Please upload a PDF file to get started.")
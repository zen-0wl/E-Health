import re
import streamlit as st
import pdfplumber
import pandas as pd
import pickle
import base64

patterns = {
        "Gender (0-M;1-F)": r"Gender.*:\s*(0|1)",
        "Blood Pressure (systolic)": r"Blood Pressure\s+([\d.]+)[/]",
        "Blood Pressure (diastolic)": r"Blood Pressure\s+[\d.]+/([\d.]+)", 
        "Heart Rate (bpm)": r"Heart Rate\s+([\d.]+)", 
        "Hemoglobin A1c (%)": r"Hemoglobin A1c\s+([\d.]+)",
        "Breathing Rate (brpm)": r"Breathing Rate\s+([\d.]+)",
        "Oxygen Saturation (%)": r"Oxygen Saturation\s+([\d.]+)", 
        "HRV SDNN (ms)": r"HRV SDNN\s+([\d.]+)",
        "RMSSD (ms)": r"RMSSD\s+([\d.]+)",
        "Recovery Ability": r"Recovery Ability.*:\s*(\d+)",
        "Mean RRI (ms)": r"Mean RRI\s+([\d.]+)",
        "Hemoglobin (g/dl)": r"Hemoglobin\s+([\d.]+)", 
        "Stress Index": r"Stress Index\s+([\d.]+)",
        "SNS Index": r"SNS Index\s+([-\d.]+)",
        "PNS Index": r"PNS Index\s+([-\d.]+)", 
        "SD1 (ms)": r"SD1\s+([\d.]+)",  
        "SD2 (ms)": r"SD2\s+([\d.]+)",
    }

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to parse extracted text based on provided format
def parse_extracted_text(text):
    parsed_data = {}
    for feature, pattern in patterns.items():
        match = re.search(pattern, text)  
        if match:
            parsed_data[feature] = float(match.group(1))
        else:
            parsed_data[feature] = None  # Set to None if not found
 
    return parsed_data

# Function to prepare input data for the model
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
    
    # Ensure the model is loaded (replace 'best_model.pkl' with your actual model file)
    with open("random_forest_model.pkl", "rb") as f:
        best_model = pickle.load(f)

    prediction = best_model.predict(input_data)
    
    predicted_disease = disease_labels[prediction[0]]

    # Display prediction result
    st.subheader("Model Prediction Result:")
    st.write(f"**Predicted Disease:** {predicted_disease}")
else:
    st.sidebar.info("Please upload a PDF file to get started.")
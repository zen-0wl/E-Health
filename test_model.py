import re
import io
import streamlit as st
import pdfplumber
import pandas as pd
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import cv2
import numpy as np 
from PIL import Image
from pdf2image import convert_from_bytes
import pickle
import base64

patterns = {
        "Gender (0-M;1-F)": r"Gender.*?:\s*(0|1)",
        "Blood Pressure (systolic)": r"Blood Pressure\s+(\d+\.\d+|\d+)/(\d+\.\d+|\d+)",
        "Blood Pressure (diastolic)": r"Blood Pressure\s+\d+\.\d+/(\d+\.\d+|\d+)",
        "Heart Rate (bpm)": r"Heart Rate\s*[:\-]?\s*(\d+\.?\d*)\s*(bpm|bpm)?", 
        "Hemoglobin A1c (%)": r"Hemoglobin Alc\s+(\d+\.\d+|\d+)\s*%",
        "Breathing Rate (brpm)": r"Breathing Rate\s*[:\-]?\s*(\d+\.?\d*)\s*(brpm|brpm)?",
        "Oxygen Saturation (%)": r"Oxygen Saturation\s+(\d+\.\d+|\d+)\s*%",
        "HRV SDNN (ms)": r"HRV SDO?NN\s+(\d+\.\d+|\d+)\s*ms",
        "RMSSD (ms)": r"RMSSD\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?",
        "Recovery Ability": r"Recovery Ability\s*[:\-]?\s*(\d+)",
        "Mean RRI (ms)": r"Mean RRI\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?",
        "Hemoglobin (g/dl)": r"Hemoglobin\s+(\d+\.\d+|\d+)\s*g/dl",
        "Stress Index": r"Stress Index\s*[:\-]?\s*(\d+\.?\d*)",
        "SNS Index": r"SNS Index\s*[:\-]?\s*(-?\d+\.?\d*)",
        "PNS Index": r"PNS Index\s*[:\-]?\s*(-?\d+\.?\d*)",
        "SD1 (ms)": r"SD1\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?",
        "SD2 (ms)": r"SD2\s*[:\-]?\s*(\d+\.?\d*)\s*(ms)?"
    }

def extract_text_from_pdf(file):
    text = ""
    file_type = file.type.lower()
    
    # If the file is a PDF
    if file_type == 'application/pdf':
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                else:
                    # Use OCR on the image if text extraction fails
                    images = convert_from_bytes(file.getvalue())
                    for img in images:
                        processed_text = extract_text_line_by_line(img)
                        text += processed_text

    elif file_type in ['image/jpeg', 'image/png', 'image/jpg']:
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
    
    return text

def extract_text_line_by_line(image):
    # Convert PIL Image to OpenCV format
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to binarize the image (improves OCR accuracy)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines (text lines) using contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1], 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours top-to-bottom by the y-coordinate
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    extracted_text = ""

    for contour in contours:
        # Get bounding box for each line
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest (line of text)
        line_img = image[y:y + h, x:x + w]

        # OCR on the line image
        line_text = pytesseract.image_to_string(line_img, config='--psm 7')
        extracted_text += line_text + "\n"

    return extracted_text

# Function to parse extracted text based on provided format
def parse_extracted_text(text):
    parsed_data = {}
    for feature, pattern in patterns.items():
        match = re.search(pattern, text)  
        if match:
            try:
                parsed_data[feature] = float(match.group(1))
            except (ValueError, TypeError):
                parsed_data[feature] = None  # Handle any conversion errors gracefully
        else:
            parsed_data[feature] = None  # Set to None if no match is found

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

st.sidebar.title("Upload File")
uploaded_file = st.sidebar.file_uploader("Drag & Drop PDF or Image here", type=["pdf", "jpg", "jpeg", "png"])

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
        if uploaded_file.type == 'application/pdf':
            show_pdf(uploaded_file)
        else:
            st.image(uploaded_file)

    with col2:
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.subheader("Extracted Text:")
        st.text_area("Text Extraction", extracted_text, height=300)

    # Parsing and model prediction
    parsed_data = parse_extracted_text(extracted_text)
    parsed_df = pd.DataFrame(list(parsed_data.items()), columns=["Disease", "Result"])
    parsed_df.index = [''] * len(parsed_df) # removes the side numberings
    
    st.subheader("Parsed Data:")
    #st.json(parsed_data)
    st.dataframe(parsed_df)

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

## Camera option 
take_photo_button = st.sidebar.button("Take Photo")

if take_photo_button:
    st.markdown(
        """
        <style>
        .stCamera {
            width: 400%;
            height: 800vh; 
        }
        </style>
        """, unsafe_allow_html=True)

    camera_input = st.camera_input("Capture Image with Camera")

    if camera_input is not None:
        image = Image.open(camera_input)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")  
        buffer.seek(0)

        with open("captured_image.png", "wb") as f:
            f.write(buffer.getvalue())
            
        st.sidebar.image(buffer, caption="Captured Image", use_column_width=True)

        # Extract text using pytesseract
        extracted_text = pytesseract.image_to_string(image)
        st.subheader("Extracted Text from Camera Image:")
        st.text_area("Extracted Text", extracted_text, height=300)

        parsed_data = parse_extracted_text(extracted_text)
        parsed_df = pd.DataFrame(list(parsed_data.items()), columns=["Disease", "Result"])
        parsed_df.index = [''] * len(parsed_df)  

        st.subheader("Parsed Data:")
        st.dataframe(parsed_df)

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
    st.sidebar.info("Click the button to open the camera.")
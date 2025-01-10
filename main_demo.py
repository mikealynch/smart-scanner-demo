import os
import easyocr
import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2
import re
import pandas as pd
import openai
from dotenv import load_dotenv
import requests
import json
import sqlite3  # For SQL database integration

# Set the environment variable before importing easyocr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

header = st.container()

load_dotenv()
api_key = os.getenv("API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Initialize session state for categorized data
if "categorized_data" not in st.session_state:
    st.session_state.categorized_data = None

# Header section for the Streamlit app
with header:
    st.title('Let\'s Scan a Business Card')

# File uploader for image input
uploaded_file = st.file_uploader("Upload a business card image for OCR analysis", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is not None and st.session_state.categorized_data is None:
    try:
        st.title('Step 1: Image Pre-processing')

        # Convert the uploaded file to a NumPy array (OpenCV format)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, 0, 180])
        upper_bound = np.array([180, 50, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found; make sure the color range includes the business card.")

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w + 2)
        h = min(image.shape[0] - y, h + 2)

        cropped_image = image[y:y+h, x:x+w]
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        compressed_image = pil_image.resize((int(pil_image.width * 0.5), int(pil_image.height * 0.5)))

        # Initialize EasyOCR reader for English and Spanish
        reader = easyocr.Reader(['en', 'es'])
        image_np = np.array(compressed_image)
        result = reader.readtext(image_np)

        text_values = [detection[1] for detection in result]
        prompt = "the information in this data set was pulled from a single business card. using this information, create valid json that only contains first name, last name, position, email, phone number, country, and company name: " + " ".join(text_values)

        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "temperature": 0.9,
            "max_tokens": 150
        }

        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            data=json.dumps(data),
            verify=False
        )

        result = response.json()
        clean_response = result['choices'][0]['text'].strip()

        while True:
            try:
                categorized_data = json.loads(clean_response)
                st.session_state.categorized_data = categorized_data
                break
            except json.JSONDecodeError:
                prompt = "convert this text to valid json: " + clean_response
                data["prompt"] = prompt
                response = requests.post(
                    "https://api.openai.com/v1/completions",
                    headers=headers,
                    data=json.dumps(data),
                    verify=False
                )
                clean_response = response.json()['choices'][0]['text'].strip()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if st.session_state.categorized_data:
    st.title("Step 4: Send Data")
    st.markdown("""
    ### Display and Edit Extracted Data

    We then display the extracted data fields in a user-friendly format. Users can review the information and make edits if necessary to ensure everything is accurate before final submission.
    """)

    updated_data = {}
    for key, value in st.session_state.categorized_data.items():
        updated_data[key] = st.text_input(key, value=value)

    if st.button("Submit to Database"):
        try:
            conn = sqlite3.connect("business_cards.db")
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS business_cards (
                first_name TEXT,
                last_name TEXT,
                position TEXT,
                email TEXT,
                phone_number TEXT,
                country TEXT,
                company_name TEXT
            )
            """)

            cursor.execute("""
            INSERT INTO business_cards (first_name, last_name, position, email, phone_number, country, company_name)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                updated_data.get("first_name", ""),
                updated_data.get("last_name", ""),
                updated_data.get("position", ""),
                updated_data.get("email", ""),
                updated_data.get("phone_number", ""),
                updated_data.get("country", ""),
                updated_data.get("company_name", "")
            ))

            conn.commit()
            conn.close()

            st.success("Data successfully submitted to the database!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

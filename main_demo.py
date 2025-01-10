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

# Helper function to convert camelCase to snake_case
def camel_to_snake(name):
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return s1.lower()

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
if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = None

# Header section for the Streamlit app
with header:
    st.title('Let\'s Scan a Business Card')

# Step 1: File Upload
st.title('Upload the Image')
uploaded_file = st.file_uploader("Upload a business card image for OCR analysis", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is not None and not st.session_state.data_uploaded:
    try:
        st.title('Image Pre-processing')

        st.write(f"Uploaded file type: {uploaded_file.type}")
        st.image(uploaded_file, caption="Uploaded File", use_container_width =True)

        # Convert the uploaded file to a NumPy array (OpenCV format)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        st.markdown("""
        ### Convert the Image from RGB to HSV
        First we decode the image into a format our OCR tool can understand. Then we convert the image from RGB (Red, Blue, Green) to HSV (Hue, Saturation, and Value) to make it easier to extract the business card region.
        """)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        st.image(hsv_image, caption="HSV Image", use_container_width =True)

        lower_bound = np.array([0, 0, 180])
        upper_bound = np.array([180, 50, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        st.markdown("""
        ### Create a Mask and Find Contours to Identify the Business Card
        1. **Contours** are the boundaries or edges of objects detected in an image. In this step, we search for the contours in the image using a mask created from color segmentation.
        2. Once contours are found, we assume that the **largest contour** corresponds to the business card. This is based on the assumption that the business card will be the largest object in the image.
        3. We then calculate the **bounding box** around this largest contour, which gives us the coordinates of the area that we believe contains the business card.
        """)

        st.image(mask, caption="Mask", use_container_width =True)

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
        

        st.markdown("""
        ### Crop and Compress the Image for Faster Processing
        To speed up processing, we compress the cropped image by resizing it to 50% of its original size. This reduces the image dimensions, making it easier and faster for EasyOCR to analyze the text, without compromising the quality too much.
        """)

        st.image(pil_image, caption="Cropped and Uncompressed Business Card", use_container_width =True)
        
        if pil_image.height > 1000:
            compressed_image = pil_image.resize((int(pil_image.width * 0.5), int(pil_image.height * 0.5)))
            st.image(compressed_image, caption="Cropped and Compressed Business Card", use_container_width =True)
            pil_image = compressed_image

        st.title('OCR Text Extraction')

        st.markdown("""
        ### Use OCR to Extract Text from the Image
        Next, we use **EasyOCR**, an optical character recognition (OCR) tool, to extract text from the image. EasyOCR is capable of recognizing text in various languages, making it ideal for business cards in different languages.
        """)

        reader = easyocr.Reader(['en', 'es'])
        image_np = np.array(pil_image)
        result = reader.readtext(image_np)

        st.write("Detected Text from the Business Card:")
        text_values = [detection[1] for detection in result]
        for text in text_values:
            st.write(f"- {text}")

        st.session_state.ocr_results = text_values

        prompt = "the information in this data set was pulled from a single business card. using this information, create valid json that only contains first name, last name, position, email, phone number, country, and company name: " + " ".join(text_values)

        st.title("Format and Process the Data")

        st.markdown("""
        #### Send Text Data to OpenAI:
        Now, we take the detected text and send it to **OpenAI's GPT model** to format the data into a **JSON structure**. The goal is to extract key fields. This step allows us to clean up the extracted information and organize it in a way that's easy to work with. Please use snake case for the keys. 
        """)

        st.markdown(f"""
        <style>
        .custom-code-block {{
            width: 100%;
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 14px;
            overflow: auto;
        }}
        </style>
        <div class="custom-code-block">{prompt}</div>
        """, unsafe_allow_html=True)

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
                categorized_data = {camel_to_snake(k): v for k, v in categorized_data.items()}
                st.session_state.categorized_data = categorized_data
                st.session_state.data_uploaded = True
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
                categorized_data = json.loads(clean_response)
                categorized_data = {camel_to_snake(k): v for k, v in categorized_data.items()}

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if st.session_state.data_uploaded and st.session_state.categorized_data:
    st.title("Review and Edit Data")
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
                first_name TEXT,
                position TEXT,
                email TEXT,
                phone_Number TEXT,
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
            st.write("Updated Data for Submission:", updated_data)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.button("View Database Contents"):
        try:
            conn = sqlite3.connect("business_cards.db")
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM business_cards")
            rows = cursor.fetchall()

            if rows:
                df = pd.DataFrame(rows, columns=["First Name", "Last Name", "Position", "Email", "Phone Number", "Country", "Company Name"])
                st.write(df)
            else:
                st.info("The database is empty.")

            conn.close()
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.button("Clear Database"):
        try:
            conn = sqlite3.connect("business_cards.db")
            cursor = conn.cursor()

            cursor.execute("DELETE FROM business_cards")
            conn.commit()
            conn.close()

            st.success("Database cleared successfully!")
        except Exception as e:
            st.error(f"An error occurred while clearing the database: {e}")

    if st.button("Create Database CSV"):
        try:
            conn = sqlite3.connect("business_cards.db")
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM business_cards")
            rows = cursor.fetchall()

            if rows:
                df = pd.DataFrame(rows, columns=["First Name", "Last Name", "Position", "Email", "Phone Number", "Country", "Company Name"])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Database CSV",
                    data=csv,
                    file_name="business_cards.csv",
                    mime="text/csv"
                )
            else:
                st.info("The database is empty. Nothing to download.")

            conn.close()
        except Exception as e:
            st.error(f"An error occurred while downloading the database: {e}")

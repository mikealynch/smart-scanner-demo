import os
import easyocr
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import re
import pandas as pd
import openai
from dotenv import load_dotenv
import requests
import json

# Set the environment variable for EasyOCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load API key from environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

# Authorization headers for OpenAI API
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Streamlit header
st.title("Let's Scan a Business Card")

# File uploader for image input
uploaded_file = st.file_uploader("Upload a business card image for OCR analysis", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    try:
        st.header("Step 1: Image Pre-processing")

        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded File", use_container_width=True)

        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        st.image(hsv_image, caption="HSV Image", use_container_width=True)

        # Define color range and create a mask
        lower_bound = np.array([0, 0, 180])  # Light color lower HSV bound
        upper_bound = np.array([180, 50, 255])  # Light color upper HSV bound
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        st.image(mask, caption="Mask", use_container_width=True)

        # Find and process contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found; make sure the color range includes the business card.")

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop and compress the image
        cropped_image = image[y:y+h, x:x+w]
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        compressed_image = pil_image.resize((int(pil_image.width * 0.5), int(pil_image.height * 0.5)))

        st.image(compressed_image, caption="Cropped and Compressed Business Card", use_container_width=True)

        st.header("Step 2: OCR")

        # Perform OCR using EasyOCR
        reader = easyocr.Reader(['en', 'es'])
        image_np = np.array(compressed_image)
        result = reader.readtext(image_np)

        # Display detected text
        text_data = [{"type": "type", "text": detection[1]} for detection in result]
        text_df = pd.DataFrame(text_data)
        st.write(text_df)

        # Prepare OpenAI prompt
        text_values = " ".join([entry['text'] for entry in text_data])
        prompt = (
            "The information in this dataset was pulled from a single business card. "
            "Using this information, create valid JSON containing first name, last name, "
            "position, email, phone number, country, and company name: "
            + text_values
        )

        st.header("Step 3: Format and Process the Data")
        st.code(prompt, language="plaintext")

        # Send text data to OpenAI
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "temperature": 0.9,
            "max_tokens": 150
        }
        response = requests.post("https://api.openai.com/v1/completions", headers=headers, data=json.dumps(data))
        clean_response = response.json()['choices'][0]['text'].strip()

        # Validate and reformat JSON
        valid_json = False
        while not valid_json:
            try:
                categorized_data = json.loads(clean_response)
                st.json(categorized_data)
                valid_json = True
            except json.JSONDecodeError:
                prompt = "Convert this text to valid JSON: " + clean_response
                data["prompt"] = prompt
                response = requests.post("https://api.openai.com/v1/completions", headers=headers, data=json.dumps(data))
                clean_response = response.json()['choices'][0]['text'].strip()

        st.header("Step 4: Display and Edit Extracted Data")
        for key, value in categorized_data.items():
            st.text_input(key.capitalize(), value=value)

        if st.button("The information above is accurate."):
            st.success("Thank you!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

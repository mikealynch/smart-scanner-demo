import os
import easyocr
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import requests
import json
from dotenv import load_dotenv

# Set the environment variable before importing EasyOCR
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Header Section
st.title("Let's Scan a Business Card")

# File Uploader
uploaded_file = st.file_uploader("Upload a business card image for OCR analysis", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    try:
        st.header("Step 1: Image Pre-processing")
        st.image(uploaded_file, caption="Uploaded File", use_container_width=True)

        # Decode uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        st.markdown("""
        ### Convert the Image from RGB to HSV
        Converting to HSV helps in isolating the business card region based on color characteristics.
        """)
        st.image(hsv_image, caption="HSV Image", use_container_width=True)

        # Create a mask to detect the business card
        lower_bound = np.array([0, 0, 180])
        upper_bound = np.array([180, 50, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        st.markdown("""
        ### Identify Contours to Detect the Business Card
        We use the largest detected contour as the business card and calculate its bounding box.
        """)
        st.image(mask, caption="Mask", use_container_width=True)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found; ensure the color range includes the business card.")

        # Extract largest contour and bounding box
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]

        # Compress and display cropped image
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        compressed_image = pil_image.resize((int(pil_image.width * 0.5), int(pil_image.height * 0.5)))

        st.markdown("""
        ### Crop and Compress the Image
        Cropping and compressing the image makes OCR analysis faster without significant loss of quality.
        """)
        st.image(compressed_image, caption="Compressed Business Card", use_container_width=True)

        # Display image dimensions
        width, height = compressed_image.size
        st.markdown(f"""
        **Compressed Image Dimensions**  
        - Width: {width}px  
        - Height: {height}px
        """)

        # OCR Step
        st.header("Step 2: OCR")
        st.markdown("""
        Using **EasyOCR** to extract text from the business card.
        """)

        reader = easyocr.Reader(['en', 'es'])
        result = reader.readtext(np.array(compressed_image))

        # Extract and display detected text
        text_data = [{"type": "detected", "text": detection[1]} for detection in result]
        text_df = pd.DataFrame(text_data)
        st.write(text_df)

        # Generate AI prompt
        text_values = " ".join([entry["text"] for entry in text_data])
        prompt = (
            "Using the following text, create a valid JSON structure with fields: "
            "first name, last name, position, email, phone number, country, and company name: "
            + text_values
        )

        # Send data to OpenAI
        st.header("Step 3: Format and Process the Data")
        st.markdown("""
        Sending the detected text to OpenAI to structure it into JSON format.
        """)
        st.code(prompt, language="python")

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

        clean_response = response.json()["choices"][0]["text"].strip()

        # Validate JSON
        st.markdown("### Validate JSON Data")
        valid_json = False

        while not valid_json:
            try:
                json_data = json.loads(clean_response)
                valid_json = True
            except json.JSONDecodeError:
                fix_prompt = f"Convert this to valid JSON: {clean_response}"
                data["prompt"] = fix_prompt
                response = requests.post(
                    "https://api.openai.com/v1/completions",
                    headers=headers,
                    data=json.dumps(data),
                    verify=False
                )
                clean_response = response.json()["choices"][0]["text"].strip()

        st.json(json_data)

        # Editable Fields
        st.header("Step 4: Review and Edit Extracted Data")
        for key, value in json_data.items():
            st.text_input(key, value=value)

        if st.button("Submit"):
            st.success("Thank you! Data submitted.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

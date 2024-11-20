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

# Set the environment variable before importing easyocr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

header = st.container()

load_dotenv()
api_key = os.getenv("API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}



# Header section for the Streamlit app
with header:
    st.title('Let\'s Scan a Business Card')

# File uploader for image input
uploaded_file = st.file_uploader("Upload a business card image for OCR analysis", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is not None:
    try:
        # Debugging: Print the type of the uploaded file to ensure it's correct
        # st.write(f"Uploaded file type: {uploaded_file.type}")
        
        # Convert the uploaded file to a NumPy array (OpenCV format)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode into OpenCV image format

        if image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        # Convert the image to HSV color space for color-based segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color range for segmentation (adjust these ranges based on card's color)
        lower_bound = np.array([0, 0, 180])   # Light color lower HSV bound
        upper_bound = np.array([180, 50, 255])  # Light color upper HSV bound

        # Create a mask for colors within the specified range
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

        # Find contours based on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found; make sure the color range includes the business card.")
            
        # Find the largest contour, assume it’s the business card
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Apply padding to include a bit more of the card's edges
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w + 2)
        h = min(image.shape[0] - y, h + 2)

        # Crop the image to the expanded bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Convert to Pillow image for saving without compression
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        
        # Compress the image by resizing it (downscale by 50%)
        compressed_image = pil_image.resize((int(pil_image.width * 0.5), int(pil_image.height * 0.5)))

        
        # Display the cropped image in Streamlit
        #st.image(compressed_image, caption="Cropped Business Card", use_container_width=True)

        # Initialize EasyOCR reader for English and Spanish
        reader = easyocr.Reader(['en', 'es'])

        # Convert the cropped image into a numpy array
        image_np = np.array(compressed_image)

        # Perform OCR with EasyOCR
        result = reader.readtext(image_np)

        # Display detected text and bounding boxes in Streamlit
        #st.write("Detected Text from the business card:")
        
        text_data = []
        
        for detection in result:
            text = detection[1]  # Extract the detected text
            bbox = detection[0]  # Extract the bounding box coordinates (optional)
            
            label = "type"
            
            
            
          
            text_data.append({
                "type": label,
                "text": text
            })
            
            
            
            
            #st.write(f'Text: {text}')
            
            #st.text_input(label, value = text)

            #st.write(f'Bounding Box: {bbox}')  # You can also show the bounding box if needed
            
        text_df = pd.DataFrame(text_data)
        #st.write(text_df)
                
                
        text_values = [entry['text'] for entry in text_data]  # Extract the 'text' from each dictionary
        prompt = "the information in this data set was pulled from a single business card. using this information, create valid json that only contains first name, last name, position, email, phone number, country, and company name: " + " ".join(text_values)
        
        #st.write(prompt)
        
        
        


        # Prepare the data for the OpenAI API request
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": prompt,
            "temperature": 0.9,
            "max_tokens": 150
        }
            
            
        # Make the API request
        response = requests.post(
            "https://api.openai.com/v1/completions",
            headers=headers,
            data=json.dumps(data),
            verify=False  # Disable SSL verification
        )
                
        result = response.json()

        clean_response = result['choices'][0]['text'].strip()
        #st.write(clean_response)
        valid_json = False

        while valid_json == False:
            try:
                json.loads(clean_response)
                #st.write(clean_response)
                valid_json = True
            except json.JSONDecodeError:
                prompt = "convert this text to valid json: " + " ".join(clean_response)
                data = {
                    "model": "gpt-3.5-turbo-instruct",
                    "prompt": prompt,
                    "temperature": 0.9,
                    "max_tokens": 150
                }
                # Make the API request
                response = requests.post(
                    "https://api.openai.com/v1/completions",
                    headers=headers,
                    data=json.dumps(data),
                    verify=False  # Disable SSL verification
                )
                        
                result = response.json()

                clean_response = result['choices'][0]['text'].strip()
                #st.write(clean_response)
                
                valid_json = False



        categorized_data = json.loads(clean_response)
        
        # Display detected text and bounding boxes in Streamlit
        st.write("Please make any necessary edits below:")
        
        for key, value in categorized_data.items():
            field_type = key
            field_text = value
            st.text_input(field_type, value = field_text)
       

        if st.button("The information above is accurate."):
            st.warning("Thanks!")

        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        

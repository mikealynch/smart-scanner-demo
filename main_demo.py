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
        st.write(f"Uploaded file type: {uploaded_file.type}")
        
        # Step 1: Convert the uploaded file to a NumPy array (OpenCV format)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Decode into OpenCV image format

        if image is None:
            raise ValueError("Failed to load image. Ensure the file is a valid image.")

        st.title('Step 1: Image Pre-processing')
        
        st.write('First we decode the image into a format our OCR tool can understand. Then we convert the image from BGR( Blue, Green, Red) to HSV (Hue, Saturation, and Value) to make it easier to extract the business card region.')
        

        # Step 2: Convert the image to HSV color space for color-based segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        st.image(hsv_image, caption="HSV Image", use_container_width=True)
        
        st.markdown("""
        ### Why Would You Use This in a Business Card Scanning App?
        
        - **Color-Based Segmentation:** If you want to extract the business card from an image, HSV helps because you can define a color range for the card (like a light-colored background) that is independent of brightness.

        - **Better Detection in Varying Light:** HSV is less sensitive to lighting variations, so if the card's color is washed out or dark, working in the HSV model helps isolate the color and perform better detection.

        """)



        # Step 3: Define color range for segmentation (adjust these ranges based on card's color)
        lower_bound = np.array([0, 0, 180])   # Light color lower HSV bound
        upper_bound = np.array([180, 50, 255])  # Light color upper HSV bound

        # Step 4: Create a mask for colors within the specified range
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        st.image(mask, caption="Mask", use_container_width=True)

        st.markdown("""
        ### Find Contours to Identify Business Card
        
        1. **Contours** are the boundaries or edges of objects detected in an image. In this step, we search for the contours in the image using the mask created from color segmentation.
        3. Once contours are found, we assume that the **largest contour** corresponds to the business card. This is based on the assumption that the business card will be the largest object in the image.
        4. We then calculate the **bounding box** around this largest contour, which gives us the coordinates of the area that we believe contains the business card.
        
        By isolating the largest contour, we ensure that we focus on the business card and exclude smaller objects or noise in the image.
        """)



        # Step 5: Find contours based on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found; make sure the color range includes the business card.")
            
        # Step 6: Find the largest contour, assume it’s the business card
        largest_contour = max(contours, key=cv2.contourArea)
        
        


        x, y, w, h = cv2.boundingRect(largest_contour)

        # Step 7: Apply padding to include a bit more of the card's edges
        x = max(0, x)
        y = max(0, y)
        w = min(image.shape[1] - x, w + 2)
        h = min(image.shape[0] - y, h + 2)

        # Step 8: Crop the image to the expanded bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Step 9: Convert to Pillow image for saving without compression
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        
        # Compress the image by resizing it (downscale by 50%)
        compressed_image = pil_image.resize((int(pil_image.width * 0.5), int(pil_image.height * 0.5)))


        st.markdown("""
        ### Compress the Image for Faster Processing
        
        To speed up processing, we compress the cropped image by resizing it to 50% of its original size. This reduces the image dimensions, making it easier and faster for EasyOCR to analyze the text, without compromising the quality too much.
        """)


        st.image(pil_image, caption="Cropped and Uncompressed Business Card", use_container_width=True)
        
        width, height = pil_image.size
                
        # Display the dimensions in Streamlit using Markdown
        st.markdown(f"""
        ### Image Dimensions
        - **Width**: {width} pixels  
        - **Height**: {height} pixels
        """)
        

        

        # Display the cropped image in Streamlit
        st.image(compressed_image, caption="Cropped and Compressed Business Card", use_container_width=True)
        
        
        width, height = compressed_image.size
                
        # Display the dimensions in Streamlit using Markdown
        st.markdown(f"""
        ### Image Dimensions
        - **Width**: {width} pixels  
        - **Height**: {height} pixels
        """)
        
        




        st.title('Step 2: OCR')
        
        st.markdown("""
        ### Use OCR to Extract Text from the Image
        
        Next, we use **EasyOCR**, an optical character recognition (OCR) tool, to extract text from the image. EasyOCR is capable of recognizing text in various languages, making it ideal for business cards in different languages.
        
        """)


        # Initialize EasyOCR reader for English and Spanish
        reader = easyocr.Reader(['en', 'es'])

        # Convert the cropped image into a numpy array
        image_np = np.array(compressed_image)

        # Perform OCR with EasyOCR
        result = reader.readtext(image_np)

        # Display detected text and bounding boxes in Streamlit
        st.write("Detected Text from the business card:")
        
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
        st.write(text_df)
                
                
        text_values = [entry['text'] for entry in text_data]  # Extract the 'text' from each dictionary
        prompt = "the information in this data set was pulled from a single business card. using this information, create valid json that only contains first name, last name, position, email, phone number, country, and company name: " + " ".join(text_values)
        
        
        
        st.title("Step 3: Format and Process the Data")
        st.markdown("""

        #### Send Text Data to OpenAI:
        Now, we take the detected text and send it to **OpenAI's GPT model** to format the data into a **JSON structure**. The goal is to extract key fields. This step allows us to clean up the extracted information and organize it in a way that's easy to work with. To achieve this I create an AI prompt and append to it the  text generated by our OCR tool:
        """)

        
        st.write(prompt)
        




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
        
        st.markdown("""
        ### Validate JSON Data
        
        After  we receive the data from **OpenAI**, we check if it’s valid **JSON**. If the data is not valid, we send it back to OpenAI to have it fixed. This ensures that the data is properly formatted before we proceed.
        """) 
        

        while valid_json == False:
            try:
                json.loads(clean_response)
                st.write(clean_response)
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
                st.write(clean_response)
                
                valid_json = False




        categorized_data = json.loads(clean_response)
        st.title("Step 4: Send Data")
        
        st.markdown("""
        ### Display and Edit Extracted Data
        
        We then display the extracted data fields in a user-friendly format. Users can review the information and make edits if necessary to ensure everything is accurate before final submission.
        
        """)
        
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
        


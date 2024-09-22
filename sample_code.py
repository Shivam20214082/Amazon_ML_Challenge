import os
import pandas as pd
import cv2
import pytesseract
from io import BytesIO
from PIL import Image
import requests
from src.utils import download_image
from src.constants import entity_unit_map

# Function to extract text from image using pytesseract
def extract_text_from_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(image_gray)
    return text.strip()

# Function to extract entity value based on entity type and units
def extract_entity_value(text, entity):
    for unit in entity_unit_map.get(entity, []):
        if unit in text.lower():
            # Extract the number preceding the unit
            import re
            match = re.search(r'(\d+\.?\d*)\s*' + unit, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {unit}"
    return ''

# Define the predictor function
def predictor(image_link, category_id, entity_name):
    # Download image
    image = download_image(image_link)
    
    # Extract text from image
    text = extract_text_from_image(image)
    
    # Extract entity value from text
    prediction = extract_entity_value(text, entity_name)
    
    return prediction

if __name__ == "__main__":
    DATASET_FOLDER = '../dataset/'
    
    # Read the test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Apply the predictor function to each row
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    # Save the predictions to a CSV file
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")

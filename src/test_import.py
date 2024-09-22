import os
import cv2
import pytesseract
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Correct path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Function to download images
def download_image(image_url, save_path):
    print(f"Downloading image from: {image_url}")  # Debugging statement
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Image saved to: {save_path}")  # Debugging statement
            return True
        else:
            print(f"Failed to download image from {image_url}")
            return False
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

# Function to perform OCR on the downloaded image
def extract_text_from_image(image_path):
    print(f"Performing OCR on: {image_path}")  # Debugging statement
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_image)
    print(f"Extracted text: {text}")  # Debugging statement
    return text

# Function to parse and clean the extracted text to get entity values
def parse_entity_value(text):
    print(f"Parsing text: {text}")  # Debugging statement
    text = text.lower()
    
    # Units from the Appendix
    entity_unit_map = {
        "width": ["centimetre", "foot", "millimetre", "metre", "inch", "yard"],
        "depth": ["centimetre", "foot", "millimetre", "metre", "inch", "yard"],
        "height": ["centimetre", "foot", "millimetre", "metre", "inch", "yard"],
        "item_weight": ["milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"],
        "maximum_weight_recommendation": ["milligram", "kilogram", "microgram", "gram", "ounce", "ton", "pound"],
        "voltage": ["millivolt", "kilovolt", "volt"],
        "wattage": ["kilowatt", "watt"],
        "item_volume": ["cubic foot", "microlitre", "cup", "fluid ounce", "centilitre", "imperial gallon", 
                        "pint", "decilitre", "litre", "millilitre", "quart", "cubic inch", "gallon"]
    }

    # Find numeric value and unit
    words = text.split()
    num_val = ""
    unit = ""
    for word in words:
        try:
            num_val = float(word)
        except ValueError:
            if word in sum(entity_unit_map.values(), []):
                unit = word
                break

    # Format the output
    if num_val and unit:
        print(f"Found entity value: {num_val} {unit}")  # Debugging statement
        return f"{num_val} {unit}"
    print("No entity value found")  # Debugging statement
    return ""

# Function to process the dataset and predict entity values
def process_dataset(dataset_path, output_file):
    print(f"Loading dataset from: {dataset_path}")  # Debugging statement
    df = pd.read_csv(dataset_path)
    print(df.head())  # Debugging statement to check if the dataset is loaded properly
    
    # Download images and perform OCR
    predictions = []
    for index, row in df.iterrows():
        image_url = row['image_link']
        save_path = f"temp_image_{index}.jpg"
        
        # Download the image
        if download_image(image_url, save_path):
            # Extract text using OCR
            extracted_text = extract_text_from_image(save_path)
            
            # Parse the text to get entity value
            prediction = parse_entity_value(extracted_text)
            predictions.append([row['index'], prediction])
            
            # Remove temp image
            os.remove(save_path)
        else:
            predictions.append([row['index'], ""])
    
    # Save predictions to the output file
    output_df = pd.DataFrame(predictions, columns=["index", "prediction"])
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")  # Debugging statement

# Main function to run the process
if __name__ == "__main__":
    # Process the dataset (change paths as needed)
    test_dataset_path = 'c:/Users/shiva/Desktop/Amazon_ML_Challenge/student_resource 3/dataset/sample_test.csv'
    output_file = 'test_out.csv'
    
    process_dataset(test_dataset_path, output_file)

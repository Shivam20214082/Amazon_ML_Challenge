import re
import os
import requests
import pandas as pd
import multiprocessing
import time
from tqdm import tqdm
from pathlib import Path
from functools import partial
from PIL import Image
from io import BytesIO
import pytesseract
import urllib.request
import constants

# Setup path to Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def common_mistake(unit):
    """Correct common unit mistakes."""
    if unit in constants.allowed_units:
        return unit
    if unit.replace('ter', 'tre') in constants.allowed_units:
        return unit.replace('ter', 'tre')
    if unit.replace('feet', 'foot') in constants.allowed_units:
        return unit.replace('feet', 'foot')
    return unit

def parse_string(s):
    """Parse a string to extract a number and unit, correcting common mistakes."""
    s_stripped = "" if s is None or str(s) == 'nan' else s.strip()
    if s_stripped == "":
        return None, None
    pattern = re.compile(r'^-?\d+(\.\d+)?\s+[a-zA-Z\s]+$')
    if not pattern.match(s_stripped):
        raise ValueError(f"Invalid format in {s}")
    parts = s_stripped.split(maxsplit=1)
    number = float(parts[0])
    unit = common_mistake(parts[1])
    if unit not in constants.allowed_units:
        raise ValueError(f"Invalid unit [{unit}] found in {s}. Allowed units: {constants.allowed_units}")
    return number, unit

def create_placeholder_image(image_save_path):
    """Create a black placeholder image when download fails."""
    try:
        placeholder_image = Image.new('RGB', (100, 100), color='black')
        placeholder_image.save(image_save_path)
        print(f"Created placeholder image at: {image_save_path}")
    except Exception as e:
        print(f"Failed to create placeholder image: {e}")

def download_image(image_link, save_folder, retries=3, delay=3):
    """Download an image from a URL, retrying if necessary."""
    if not isinstance(image_link, str):
        print(f"Invalid image link: {image_link}")
        return

    filename = Path(image_link).name
    image_save_path = os.path.join(save_folder, filename)

    if os.path.exists(image_save_path):
        print(f"Image already exists: {image_save_path}")
        return

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(image_link, image_save_path)
            print(f"Downloaded image: {image_save_path}")
            return
        except Exception as e:
            print(f"Failed to download image {image_link} (Attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    
    create_placeholder_image(image_save_path)  # Create a placeholder if all retries fail

def download_images(image_links, download_folder, allow_multiprocessing=True):
    """Download multiple images from a list of URLs."""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
        print(f"Created download folder: {download_folder}")

    if allow_multiprocessing:
        download_image_partial = partial(
            download_image, save_folder=download_folder, retries=3, delay=3
        )

        with multiprocessing.Pool(64) as pool:
            list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)))
    else:
        for image_link in tqdm(image_links, total=len(image_links)):
            download_image(image_link, save_folder=download_folder, retries=3, delay=3)

def extract_text_from_image(image_path):
    """Use OCR to extract text from an image."""
    try:
        with Image.open(image_path) as image:
            text = pytesseract.image_to_string(image)
            return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def parse_text(text, keyword):
    """Parse text to find specific details."""
    pattern = re.compile(rf'(\d+(\.\d+)?)\s*{keyword}', re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return float(match.group(1))
    return None

def predictor(image_link, entity_name, download_folder):
    """Predictor function to extract details from an image."""
    # Create the download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    # Download image
    filename = Path(image_link).name
    image_save_path = os.path.join(download_folder, filename)
    download_image(image_link, download_folder)
    
    if not os.path.exists(image_save_path):
        return "image_download_failed"
    
    # Extract text from image
    text = extract_text_from_image(image_save_path)
    if not text:
        return "text_extraction_failed"
    
    # Extract relevant information based on the entity name
    if entity_name.lower() == "weight":
        value = parse_text(text, "weight")
    elif entity_name.lower() == "height":
        value = parse_text(text, "height")
    else:
        return "unknown_entity"
    
    if value is not None:
        return value
    else:
        return "information_not_found"

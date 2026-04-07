import cv2
import easyocr
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
import numpy as np
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OCR Engines
logger.info("Initializing EasyOCR...")
easyocr_reader = easyocr.Reader(['en'], gpu=False)

logger.info("Initializing PaddleOCR...")
paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

# Function to load image using OpenCV
def load_image(image_path):
    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image from path: {image_path}")
    return image

# Function to run all OCR engines
def run_ocr(image_path):
    image = load_image(image_path)

    # EasyOCR
    logger.info("Running EasyOCR...")
    easyocr_result = easyocr_reader.readtext(image_path, detail=0)

    # PaddleOCR
    logger.info("Running PaddleOCR...")
    paddle_result = paddleocr_reader.ocr(image_path, cls=True)
    paddle_text = [line[1][0] for line in paddle_result[0]]

    # Tesseract (requires PIL Image)
    logger.info("Running Tesseract...")
    tesseract_text = pytesseract.image_to_string(Image.open(image_path))

    return {
        "easyocr_result": easyocr_result,
        "paddleocr_result": paddle_text,
        "tesseract_result": tesseract_text
    }

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR on an image using EasyOCR, PaddleOCR, and Tesseract.")
    parser.add_argument("image", help="OCR_Demo/Product Details Images/1.jpg")
    args = parser.parse_args()

    results = run_ocr(args.image)

    print("\n--- OCR Results ---")
    print("\nEasyOCR:")
    for line in results['easyocr_result']:
        print(line)

    print("\nPaddleOCR:")
    for line in results['paddleocr_result']:
        print(line)

    print("\nTesseract:")
    print(results['tesseract_result'])

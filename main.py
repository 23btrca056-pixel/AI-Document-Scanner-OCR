import cv2
import numpy as np
from PIL import Image
import pytesseract

# Set tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def enhance_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img, thresh

def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text

def main():
    input_path = "sample.jpg"  # Replace with your document image
    original, processed = enhance_image(input_path)

    cv2.imwrite("processed_output.png", processed)

    text = extract_text(processed)
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("OCR Completed!")
    print("Extracted Text:
", text)

if __name__ == "__main__":
    main()

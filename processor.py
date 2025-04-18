import cv2 # type: ignore
import numpy as np # type: ignore
from pytesseract import pytesseract # type: ignore
import re
import os

# Folder with input images
image_folder = "uploads"

# Optional: Save OCR results
results_output_file = "ocr_results.csv"

# OCR config: whitelist A-Z and 0-9 (tuned for single line)
custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'

# Regex
pattern = r'[A-Z]\d{5}'

# TODO: From the image, detect the blue rectangle and crop the image to that rectangle
def blue_rectangle(image):
    # crops the image detected with blue rectangle
    return image

def extract_text(image):
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image, config=custom_config)
    cleaned = text.strip().replace(" ", "").replace("\n", "")
    match = re.search(pattern, cleaned)
    if match:
        return match.group(0)
    else:
        return None

def preprocess_image(image):
    # Preprocess the image
    # Check each step if you can find parking pass number in the intermediate steps

    # Normalization
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    img = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

    # Noise Removal
    noise_removal = cv2.fastNlMeansDenoisingColored(img, 10, 7, 15)
    text = extract_text(noise_removal)
    if text:
        return text
    
    # Invert the image which basically switches the black and white pixels
    invert_image = cv2.bitwise_not(noise_removal)
    text = extract_text(invert_image)
    if text:
        return text
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(invert_image, cv2.COLOR_BGR2GRAY)
    text = extract_text(gray)
    if text:
        return text
    
    thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = extract_text(thresh)
    if text:
        return text
    
    inverted_thresh = cv2.bitwise_not(thresh)
    text = extract_text(inverted_thresh)
    if text:
        return text
    
    equilized = cv2.equalizeHist(inverted_thresh)
    blurred = cv2.GaussianBlur(equilized, (5, 5), 0)
    text = extract_text(blurred)
    if text:
        return text
    
    threshed = cv2.adaptiveThreshold(equilized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(threshed, -1, kernel)
    text = extract_text(sharpened)
    if text:
        return text
    
    return None

def process_image(image_path):
    try:
        image = cv2.imread(image_path)

        # blue_rectangle(image) function should be defined here to detect blue rectangles
        # For now, let's assume it returns a cropped image of detected rectangles
        blue = blue_rectangle(image)

        # Crop the bottom half of image
        height, width = blue.shape[:2]
        bottom_half = blue[height//2:height, 0:width]

        # Preprocess the image + text extraction

        text = preprocess_image(bottom_half)
        return text
    except Exception as e:
        return f"Error: {e}"

def main():
    results = []

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            full_path = os.path.join(image_folder, filename)
            result = process_image(full_path)
            results.append((filename, result))
            if result is None:
                print(f"{filename}: No parking pass number detected.")
            else:
                print(f"{filename}: Z{result}")

    # Save results to CSV
    with open(results_output_file, "w") as f:
        f.write("filename,ocr_result\n")
        for name, text in results:
            if text is None:
                f.write(f"{name},{text}\n")
            else:
                f.write(f"{name},Z{text}\n")

if __name__ == "__main__":
    main()  # Call the main function to start the process
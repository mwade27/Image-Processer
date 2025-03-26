import cv2
import numpy as np
from pytesseract import pytesseract

# Load an image from file
image = cv2.imread('uploads/test1.jpg')  

## PreProdessing the Image 
# Convert the image to grayscale

# Invert the image which basically switches the black and white pixels
invert_image = cv2.bitwise_not(image)
#cv2.imwrite('uploads/inverted.jpg', invert_image)

# Convert the image to grayscale
gray = cv2.cvtColor(invert_image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 220, 200, cv2.THRESH_BINARY)[1]
cv2.imwrite('uploads/thresh.jpg', thresh)

inverted_thresh = cv2.bitwise_not(thresh)
cv2.imwrite('uploads/inverted_thresh.jpg', inverted_thresh)


#equilized = cv2.equalizeHist(gray)




# Apply GaussianBlur to the image
#blurred = cv2.GaussianBlur(equilized, (5, 5), 0)
# Set the image to black and white depending on each pixel intensity
#thresh = cv2.adaptiveThreshold(equilized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

kernel = np.array([[-1, -1, -1], 
                 [-1,  9, -1], 
                  [-1, -1, -1]])

#sharpened = cv2.filter2D(thresh, -1, kernel)

""" oem is the OCR Engine Mode
    3 is for Default, 
    which automactically uses the best OCR engine
    psm is the Page Segmentation Mode
    the 6 is for Assume a single uniform block of text
    psm 7 is for Treating the image as a single text line
"""
custom_config = r'--oem 3 --psm 6'
# Use pytesseract to extract text
text = pytesseract.image_to_string(inverted_thresh, config=custom_config)

print("Extracted Text: ")
print(text)
# Display the preprocessed image in a window
print("\nEnd of Text")


cv2.imshow('Image', inverted_thresh)

# Display the non preprocessed image in a window
# cv2.imshow('Image2', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
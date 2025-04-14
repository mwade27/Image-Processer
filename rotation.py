from PIL import Image

# Load the image
img = Image.open('uploads/test1.jpg')

# Rotate the image (e.g., 90 degrees counter-clockwise)
rotated_img = img.rotate(270, expand=True)  # use expand=True to fit the whole image

# Save the rotated image
rotated_img.save("rotated_test1.jpg")

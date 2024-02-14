import cv2
import os

# Define input and output directories
input_dir = "dataimg"
output_dir = "imgL"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in the input directory
files = os.listdir(input_dir)

for file in files:
    # Check if the file is an image (you can add more extensions if needed)
    if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        # Read the image
        img = cv2.imread(os.path.join(input_dir, file))

        if img is not None:
            # Resize the image to 150x150
            img_resized = cv2.resize(img, (150, 150))

            # Save the resized image to the output directory
            cv2.imwrite(os.path.join(output_dir, file), img_resized)

print("Images resized and saved to", output_dir)

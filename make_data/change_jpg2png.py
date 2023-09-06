from PIL import Image
import os

# Path to the folder containing the JPEG image
folder_path = "./DATA/train/fake_images/"

# List all the files in the folder
file_list = os.listdir(folder_path)
print(len(file_list))
# Iterate through the files in the folder
for file_name in file_list:
    if file_name.endswith(".jpg"):
        # Open the JPEG image
        image_path = os.path.join(folder_path, file_name)
        img = Image.open(image_path)
        
        # Convert the image to PNG format
        png_path = os.path.splitext(image_path)[0] + ".png"
        img.save(png_path, "PNG")
        
        print(f"Converted {file_name} to PNG and saved as {png_path}")
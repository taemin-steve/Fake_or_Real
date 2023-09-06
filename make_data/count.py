import os
from PIL import Image

def count_images_in_folder(folder_path):
    count = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed

    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            _, extension = os.path.splitext(filename)
            if extension.lower() in image_extensions:
                try:
                    Image.open(os.path.join(folder_path, filename))
                    count += 1
                except (IOError, SyntaxError):
                    # Skip files that are not valid images
                    pass

    return count

# Provide the path to the folder containing the images
folder_path = './Stable/'
image_count = count_images_in_folder(folder_path)
print(f"Number of images in folder: {image_count}")
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# Read the CSV file
df = pd.read_csv('./make_data/diffusion_prompts.csv')

# Get the image URLs from the 'url' column
image_urls = df['url'][20000:]
count = 0 
# Iterate over the image URLs and save the images
for i, url in enumerate(image_urls):
    # Send a GET request to download the image
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Load the image from the response content
        image = Image.open(BytesIO(response.content))
        
        # Save the image with a unique name
        image.save(f'./Data/train/fake_images/SDNewfake_{i}.png')
        count += 1
        if count > 10000:
            break
        # print(f"Image {i} saved successfully.")
    else:
        print(f"Failed to download image from URL: {url}")
        print(count)
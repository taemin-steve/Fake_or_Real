import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# Read the CSV file
df = pd.read_csv('./make_data/midjourney_2022_250k.csv')

# Get the image URLs from the 'img_url' column
image_urls = df['img_url']

# Iterate over the image URLs and load/save the images
for i, url in enumerate(image_urls):
    # Construct the complete URL
    complete_url = 'https://cdn.discordapp.com/attachments/' + url

    # Send a GET request to download the image
    response = requests.get(complete_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Load the image from the response content
        image = Image.open(BytesIO(response.content))
        
        # Save the image with a unique name
        image.save(f'./newdata/fakeimage_{i}.jpg')
    else:
        print(f"Failed to download image from URL: {complete_url}")
    if i > 4000:
        break
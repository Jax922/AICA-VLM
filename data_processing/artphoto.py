import os
import pandas as pd

# Set the path to the folder containing images
image_folder = "testImages_artphoto"

# Define the emotion labels
emotion_labels = ['amusement', 'anger', 'awe', 'content', 'disgust', 'excitement', 'fear', 'sad']

# Get all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Initialize a list to store image information
image_data = []

# Iterate through each image file
for image_file in image_files:
    # Get the image name
    image_name = image_file  # The original name of the image

    # Set the default emotion label to None
    emotion_cat = None

    # Check if the image name contains an emotion label
    for label in emotion_labels:
        if label in image_name.lower():  # Check if the emotion label is in the image name
            emotion_cat = label
            break  # Exit the loop once a matching emotion label is found

    # Set the folder and other fields
    img_folder = "Artphoto"  # All images are in the "Artphoto" folder
    emotion_v = None
    emotion_a = None
    dataset_source = "testImage_artphoto"

    # Add the image information to the list
    image_data.append([image_name, img_folder, emotion_cat, emotion_v, emotion_a, dataset_source])

# Convert the image information to a DataFrame
image_df = pd.DataFrame(image_data,
                        columns=['img_name', 'img_folder', 'emotion_cat', 'emotion_v', 'emotion_a', 'dataset_source'])

# Save the data as a CSV file
image_df.to_csv('artphoto.csv', index=False)

print("Image information has been successfully saved to 'artphoto.csv'.")

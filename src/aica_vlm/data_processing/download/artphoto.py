# ArtPhoto dataset
# https://www.imageemotion.org

import os
import shutil

import pandas as pd


def process_artphoto_data(image_folder, output_csv_path, output_dir):
    """
    Process the Artphoto dataset, extract emotion categories based on image names,
    and organize the images into the target directory.

    Parameters:
    - image_folder: Path to the folder containing the images.
    - output_csv_path: Path to save the CSV file containing image information.
    - output_dir: Directory to save the organized images.
    """
    # Define the emotion labels
    emotion_labels = [
        "amusement",
        "anger",
        "awe",
        "content",
        "disgust",
        "excitement",
        "fear",
        "sad",
    ]

    # Get all image files in the folder
    image_files = [
        f
        for f in os.listdir(image_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    # Initialize a list to store image information
    image_data = []

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each image file
    for image_file in image_files:
        # Get the image name
        image_name = image_file  # The original name of the image

        # Set the default emotion label to None
        emotion_cat = None

        # Check if the image name contains an emotion label
        for label in emotion_labels:
            if (
                label in image_name.lower()
            ):  # Check if the emotion label is in the image name
                emotion_cat = label
                break  # Exit the loop once a matching emotion label is found

        # Set the folder and other fields
        img_folder = "Artphoto"  # All images are in the "Artphoto" folder
        emotion_v = None
        emotion_a = None
        dataset_source = "testImage_artphoto"

        # Add the image information to the list
        image_data.append(
            [image_name, img_folder, emotion_cat, emotion_v, emotion_a, dataset_source]
        )

        # Copy image to the target directory
        original_img_path = os.path.join(image_folder, image_name)
        if not os.path.exists(original_img_path):
            print(f"[Missing] {original_img_path}")
            continue

        target_img_path = os.path.join(output_dir, image_name)
        shutil.copy2(original_img_path, target_img_path)

    # Convert the image information to a DataFrame
    image_df = pd.DataFrame(
        image_data,
        columns=[
            "img_name",
            "img_folder",
            "emotion_cat",
            "emotion_v",
            "emotion_a",
            "dataset_source",
        ],
    )

    # Save the data as a CSV file
    image_df.to_csv(output_csv_path, index=False)

    print(f"Image information has been successfully saved to '{output_csv_path}'")


if __name__ == "__main__":
    image_folder = r"C:\Users\yu_da\Desktop\VLM data\download\testImages_artphoto"  # Specify the actual image directory
    output_csv_path = "./datasets/Artphoto/artphoto.csv"  # Path to save the CSV file
    output_dir = (
        "./datasets/Artphoto/processed_images"  # Directory to store processed images
    )

    # Run the function to process the dataset
    process_artphoto_data(image_folder, output_csv_path, output_dir)

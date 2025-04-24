# Abstract
# https://www.imageemotion.org

import os
import shutil

import pandas as pd


def process_abstract_data(input_csv_path, output_csv_path, image_folder, output_dir):
    """
    Process the Abstract dataset, extract emotion categories from CSV,
    and organize images into the target directory.

    Parameters:
    - input_csv_path: Path to the input CSV file.
    - output_csv_path: Path to save the processed CSV file.
    - image_folder: Directory containing the images.
    - output_dir: Directory to save the organized images.
    """
    # Load the dataset
    df = pd.read_csv(input_csv_path)
    print(df.columns)

    emotion_columns = [
        "Amusement",
        "Anger",
        "Awe",
        "Content",
        "Disgust",
        "Excitement",
        "Fear",
        "Sad",
    ]

    # Initialize a list to store the processed data
    processed_data = []

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Find the emotion with the maximum value
        emotion_values = row[emotion_columns]
        max_emotion = emotion_values.idxmax()
        max_value = emotion_values.max()

        # Prepare the row for the new CSV file
        img_name = row.iloc[0].strip("'")  # Remove leading and trailing single quotes
        img_folder = "testImages_abstract"  # All images are in the 'Abstract' folder
        emotion_v = None  # No values provided for 'emotion_v'
        emotion_a = None  # No values provided for 'emotion_a'
        dataset_source = "abstract_dataset"  # New column to indicate the dataset source

        # Append the processed row
        processed_data.append(
            [img_name, img_folder, max_emotion, emotion_v, emotion_a, dataset_source]
        )

        # Copy image to the target directory
        original_img_path = os.path.join(image_folder, img_name)
        if not os.path.exists(original_img_path):
            print(f"[Missing] {original_img_path}")
            continue

        target_img_path = os.path.join(output_dir, img_name)
        shutil.copy2(original_img_path, target_img_path)

    # Convert the processed data into a DataFrame
    processed_df = pd.DataFrame(
        processed_data,
        columns=[
            "img_name",
            "img_folder",
            "emotion_cat",
            "emotion_v",
            "emotion_a",
            "dataset_source",
        ],
    )

    # Save the new DataFrame to a CSV file
    processed_df.to_csv(output_csv_path, index=False)

    print(f"CSV file '{output_csv_path}' has been created.")


if __name__ == "__main__":
    input_csv_path = "ABSTRACT_groundTruth.csv"  # Path to input CSV file
    output_csv_path = (
        "./datasets/Abstract/Abstract.csv"  # Path to save the processed CSV file
    )
    image_folder = r"C:\Users\yu_da\Desktop\VLM data\download\testImages_abstract"  # Path to the folder containing images
    output_dir = (
        "./datasets/Abstract/processed_images"  # Directory to store processed images
    )

    # Run the function to process the dataset
    process_abstract_data(input_csv_path, output_csv_path, image_folder, output_dir)

# FlickrLDL dataset
# https://github.com/sherleens/EmotionDistributionLearning/blob/57eb79073e7750132e464292ac890b0dc4e02db2/README.md#download-dataset-lmdb

import os
import shutil

import h5py
import pandas as pd


def process_flickrldl_data(
    mat_path, output_csv_path, final_csv_path, image_root_dir, output_dir
):
    """
    Process the Flickr LDL dataset, extract annotation data, and organize the images.

    Parameters:
    - mat_path: Path to the .mat file.
    - output_csv_path: Path to save intermediate CSV containing emotion votes.
    - final_csv_path: Path to save the final CSV with processed annotations.
    - image_root_dir: Root directory containing the images.
    - output_dir: Directory to save the organized images.
    """
    # Check if the file path exists
    if not os.path.isfile(mat_path):
        print(f"Error: The file {mat_path} does not exist!")
        return

    # Open the .mat file
    with h5py.File(mat_path, "r") as f:
        # Read 'vote' data
        vote = f["vote"][:]
        # Transpose the vote data so that each row represents an image and each column represents a category of emotion
        vote = (
            vote.T
        )  # After transpose, vote shape is (11150, 8), each row corresponds to an image, each column corresponds to an emotion category

        # Create DataFrame for the emotion categories
        data = {
            "anger": vote[:, 0],
            "amusement": vote[:, 1],
            "awe": vote[:, 2],
            "contentment": vote[:, 3],
            "disgust": vote[:, 4],
            "excitement": vote[:, 5],
            "fear": vote[:, 6],
            "sadness": vote[:, 7],
        }

        # Create DataFrame for vote data and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_csv_path, index=False)
        print(f"Intermediate data has been saved to: {output_csv_path}")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the intermediate CSV data
    df = pd.read_csv(output_csv_path)

    # Emotion label list
    emotion_labels = [
        "anger",
        "amusement",
        "awe",
        "contentment",
        "disgust",
        "excitement",
        "fear",
        "sadness",
    ]

    # Create an empty DataFrame to store processed data
    processed_data = {
        "img_name": [f"{index + 1}.jpg" for index in range(len(df))],
        "img_folder": ["LJCAI7"] * len(df),  # Fixed folder name 'LJCAI7'
        "emotion_cat": [],
        "emotion_v": [""] * len(df),  # Leave empty
        "emotion_a": [""] * len(df),  # Leave empty
        "dataset_source": ["Flickr_LDL"] * len(df),
    }

    # Iterate through each row to find the emotion category corresponding to the highest probability
    for index, row in df.iterrows():
        vote = row[emotion_labels].values
        max_index = vote.argmax()  # Get the index of the maximum value
        emotion_cat = emotion_labels[max_index]  # Get the corresponding emotion label

        # Append to processed data
        processed_data["emotion_cat"].append(emotion_cat)

        # Copy images to the target folder
        img_name = f"{index + 1}.jpg"
        original_img_path = os.path.join(image_root_dir, img_name)

        if not os.path.exists(original_img_path):
            print(f"[Missing] {original_img_path}")
            continue

        target_img_path = os.path.join(output_dir, img_name)
        shutil.copy2(original_img_path, target_img_path)

    # Create a new DataFrame for the processed data
    processed_df = pd.DataFrame(processed_data)

    # Save the processed data to the final CSV file
    processed_df.to_csv(final_csv_path, index=False)
    print(f"Data has been processed and saved to: {final_csv_path}")


if __name__ == "__main__":
    mat_path = r"./datasets/Flickr_LDL/flickrldl_config.mat"
    output_csv_path = r"./datasets/Flickr_LDL/output.csv"
    final_csv_path = r"./datasets/Flickr_LDL/Flickr_LDL_annotation.csv"
    image_root_dir = (
        r"./datasets/Flickr_LDL/images"  # Specify the actual image directory
    )
    output_dir = r"./datasets/Flickr_LDL/processed_images"  # Specify the directory to store processed images

    # Run the function to process the dataset
    process_flickrldl_data(
        mat_path, output_csv_path, final_csv_path, image_root_dir, output_dir
    )

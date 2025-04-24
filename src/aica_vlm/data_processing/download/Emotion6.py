# Emotion6 dataset
# http://chenlab.ece.cornell.edu/downloads.html

import csv
import os

import pandas as pd


def convert_txt_to_csv(input_file, output_file):
    """
    Convert a tab-delimited txt file to CSV, excluding any columns with 'prob.' in the name

    Args:
        input_file (str): Path to the input txt file
        output_file (str): Path to the output csv file
    """
    # Read the input file
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse the header line
    header_line = lines[0].strip()
    headers = header_line.split("\t")

    # Identify the indices of columns to keep (exclude probability columns)
    keep_column_indices = []
    for i, header in enumerate(headers):
        if "prob." not in header:
            keep_column_indices.append(i)

    # Create the CSV content
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header row (only for columns we're keeping)
        new_headers = [headers[i] for i in keep_column_indices]
        csv_writer.writerow(new_headers)

        # Process each data line
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                continue

            values = line.split("\t")

            # Select only the values for columns we want to keep
            keep_values = [values[i] for i in keep_column_indices]

            # Write to CSV
            csv_writer.writerow(keep_values)

    print(f"Successfully converted {input_file} to {output_file}!")


def process_csv(input_file, output_file):
    """
    Processes a CSV file by replacing all '/' with '_' in the img_name column.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the output CSV file
    """
    # Read the ground_truth.csv file
    df = pd.read_csv(input_file)

    # Create a new DataFrame
    new_df = pd.DataFrame()

    # Process column data
    new_df["img_name"] = df["[image_filename]"]
    # Add this line to replace '/' with '_' in img_name column
    new_df["img_name"] = new_df["img_name"].apply(lambda x: x.replace("/", "_"))
    new_df["img_folder"] = "Emotion6_images"
    # Extract emotion_cat (extract the word before '/' from image_filename)
    new_df["emotion_cat"] = df["[image_filename]"].apply(lambda x: x.split("/")[0])

    # Normalize valence and arousal from 1-9 to -1 to 1
    # Normalization formula: (value - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
    # For 1-9 range mapping to -1 to 1: (value - 1) / (9 - 1) * (1 - (-1)) + (-1) = (value - 1) / 4 - 1
    new_df["emotion_v"] = df["[valence]"].apply(lambda x: (x - 1) / 4 - 1)
    new_df["emotion_a"] = df["[arousal]"].apply(lambda x: (x - 1) / 4 - 1)

    # Add dataset_source column
    new_df["dataset_source"] = "Emotion6_dataset"

    # Save the results to a new CSV file
    new_df.to_csv(output_file, index=False)


# Call the function
if __name__ == "__main__":
    convert_txt_to_csv(
        "C:\\Users\\29835\\Desktop\\VLM-EQ-main\\VLM-EQ-main\\datasets\\Emotion6_dataset\\ground_truth.txt",
        "ground_truth.csv",
    )
    process_csv("ground_truth.csv", "Emotion6_dataset_annotations.csv")

    print("CSV processing completed successfully.")

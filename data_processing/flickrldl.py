import h5py
import pandas as pd
import os

# File paths
file_path = "./datasets/Flickr_LDL/flickrldl_config.mat"
output_csv_path ="./datasets/Flickr_LDL/output.csv"
final_csv_path ="./datasets/Flickr_LDL/Flickr_LDL_annotation.csv"

# Check if the file path exists
if not os.path.isfile(file_path):
    print(f"Error: The file {file_path} does not exist!")
    exit()

# Open the .mat file
with h5py.File(file_path, 'r') as f:
    # Read 'vote' data

    vote = f['vote'][:]

    # Print the shape of the vote data
    print(f"Shape of vote data: {vote.shape}")

    # Transpose the vote data so that each row represents an image and each column represents a category of emotion
    vote = vote.T  # After transpose, vote shape is (11150, 8), each row corresponds to an image, each column corresponds to an emotion category

    # Create a DataFrame to store the data
    data = {
        'anger': vote[:, 0],
        'amusement': vote[:, 1],
        'awe': vote[:, 2],
        'contentment': vote[:, 3],
        'disgust': vote[:, 4],
        'excitement': vote[:, 5],
        'fear': vote[:, 6],
        'sadness': vote[:, 7],
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save the intermediate CSV file
    df.to_csv(output_csv_path, index=False)

print(f"Intermediate data has been saved to: {output_csv_path}")

# Load the intermediate CSV data
df = pd.read_csv(output_csv_path)

# Emotion category list
emotion_labels = ['anger', 'amusement', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

# Create an empty DataFrame to store processed data
processed_data = {
    'img_name': [f"{index + 1}.jpg" for index in range(len(df))],
    'img_folder': ['LJCAI7'] * len(df),  # Fixed folder name 'LJCAI7'
    'emotiona_cat': [],
    'emotion_v': [''] * len(df),  # Leave empty
    'emotion_a': [''] * len(df),  # Leave empty
    'dataset_source': ['Flickr_LDL'] * len(df)
}

# Iterate through each row and find the emotion category with the highest probability
for index, row in df.iterrows():
    # Get the current row's vote data
    vote = row[emotion_labels].values
    # Find the emotion category with the highest probability
    max_index = vote.argmax()  # Get the index of the maximum value
    emotion_cat = emotion_labels[max_index]  # Get the corresponding emotion label

    # Append to the processed data
    processed_data['emotiona_cat'].append(emotion_cat)

# Create a new DataFrame
processed_df = pd.DataFrame(processed_data)

# Save the processed data to the final CSV file
processed_df.to_csv(final_csv_path, index=False)


print(f"Data has been processed and saved to: {final_csv_path}")

import h5py
import pandas as pd

# File paths
file_path = "./datasets/Twitter_LDL/twitterldl_config.mat"
output_csv_path = "./datasets/Twitter_LDL/output.csv"
final_csv_path = "./datasets/Twitter_LDL/Twitter_LDL_annotation.csv"

# Open the .mat file
with h5py.File(file_path, 'r') as f:

    vote = f['vote'][:]

    vote = vote.T  # Transposed vote shape is (11150, 8), each row corresponds to an image, and each column corresponds to an emotion category

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

    # Save as CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Data has been saved to: {output_csv_path}")

df = pd.read_csv(output_csv_path)

# Emotion label list
emotion_labels = ['anger', 'amusement', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

# Create an empty DataFrame to store the processed data
processed_data = {
    'img_name': [f"{index + 1}.jpg" for index in range(len(df))],  # Create each image name in the format '1.jpg', '2.jpg', ...
    'img_folder': ['LJCAI7'] * len(df),  # Fixed as 'LJCAI7'
    'emotiona_cat': [],
    'emotion_v': [''] * len(df),  # Leave empty
    'emotion_a': [''] * len(df),  # Leave empty
    'dataset_source': ['Twitter_LDL'] * len(df)  # Fixed as 'Twitter_LDL'
}

# Iterate through each row to find the emotion category corresponding to the highest probability
for index, row in df.iterrows():
    # Get the vote data for the current row
    vote = row[emotion_labels].values
    # Find the emotion category with the highest probability
    max_index = vote.argmax()  # Get the index of the maximum value
    emotion_cat = emotion_labels[max_index]  # Get the corresponding emotion label

    # Add to processed_data
    processed_data['emotiona_cat'].append(emotion_cat)

# Create a new DataFrame
processed_df = pd.DataFrame(processed_data)

# Save the processed data to a new CSV file
processed_df.to_csv(final_csv_path, index=False)

print(f"Data has been processed and saved to: {final_csv_path}")

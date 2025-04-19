import pandas as pd

# Load the dataset (replace 'your_file.csv' with the actual path to your CSV file)
df = pd.read_csv('ABSTRACT_groundTruth.csv')

print(df.columns)

emotion_columns = ['Amusement', 'Anger', 'Awe', 'Content', 'Disgust', 'Excitement', 'Fear', 'Sad']

# Initialize a list to store the processed data
processed_data = []

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    # Find the emotion with the maximum value
    emotion_values = row[emotion_columns]
    max_emotion = emotion_values.idxmax()
    max_value = emotion_values.max()

    # Prepare the row for the new CSV file
    img_name =row.iloc[0]
    img_folder = "testImages_abstract"  # All images are in the 'Abstract' folder
    emotion_v = None  # No values provided for 'emotion_v'
    emotion_a = None  # No values provided for 'emotion_a'
    dataset_source = "abstract_dataset"  # New column to indicate the dataset source

    # Append the processed row
    processed_data.append([img_name, img_folder, max_emotion, emotion_v, emotion_a, dataset_source])

# Convert the processed data into a DataFrame
processed_df = pd.DataFrame(processed_data, columns=['img_name', 'img_folder', 'emotion_cat', 'emotion_v', 'emotion_a',
                                                     'dataset_source'])

# Save the new DataFrame to a CSV file
processed_df.to_csv('Abstract.csv', index=False)

print("CSV file 'Abstract' has been created.")


import os
import csv
import re

# Specify the path to the image folder
image_folder = "FI_dataset_images"
# Specify the path to the output CSV file
output_csv = "FI_dataset_annotations.csv"

# Create a CSV file
with open(output_csv, 'w', newline='') as csvfile:
    # Define the CSV header
    fieldnames = ['img_name', 'img_folder', 'emotion_cat', 'emotion_v', 'emotion_a', 'dataset_source']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Iterate through all the images in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.jpg'):
            # Use regular expressions to extract the label from the filename (the content before the first underscore)
            match = re.match(r"([^_]+)_", filename)
            
            if match:
                emotion_category = match.group(1)
                
                # Create and write a line of data
                writer.writerow({
                    'img_name': filename,
                    'img_folder': 'emotion_dataset_images',
                    'emotion_cat': emotion_category,
                    'emotion_v': '',  
                    'emotion_a': '', 
                    'dataset_source': 'FI_dataset'
                })

print(f"CSV file has been created : {output_csv}")
print(f"CSV processing completed successfully.")
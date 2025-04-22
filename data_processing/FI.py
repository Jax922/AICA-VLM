import os
import csv
import re

def extract_emotion_category(filename):
    """Extract emotion category from filename using regex."""
    match = re.match(r"([^_]+)_", filename)
    return match.group(1) if match else None

def create_row_data(filename, emotion_category):
    """Create a dictionary representing a row of data."""
    return {
        'img_name': filename,
        'img_folder': 'emotion_dataset_images',
        'emotion_cat': emotion_category,
        'emotion_v': '',  
        'emotion_a': '', 
        'dataset_source': 'FI_dataset'
    }

def process_images(image_folder, output_csv):
    """Process images in the folder and write data to CSV."""
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
                emotion_category = extract_emotion_category(filename)
                
                if emotion_category:
                    # Create and write a line of data
                    row_data = create_row_data(filename, emotion_category)
                    writer.writerow(row_data)
    
    print(f"CSV file has been created : {output_csv}")
    print(f"CSV processing completed successfully.")

def main():
    """Main function to run the script."""
    # Specify the path to the image folder
    image_folder = "C:\\Users\\29835\\Desktop\\VLM-EQ-main\\VLM-EQ-main\\datasets\\FI_dataset\\FI_dataset_images"
    # Specify the path to the output CSV file
    output_csv = "FI_dataset_annotations.csv"
    
    process_images(image_folder, output_csv)

if __name__ == "__main__":
    main()
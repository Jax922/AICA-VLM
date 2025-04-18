import csv
import os

def convert_txt_to_csv(input_file, output_file):
    """
    Convert a tab-delimited txt file to CSV, excluding any columns with 'prob.' in the name
    
    Args:
        input_file (str): Path to the input txt file
        output_file (str): Path to the output csv file
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse the header line
    header_line = lines[0].strip()
    headers = header_line.split('\t')
    
    # Identify the indices of columns to keep (exclude probability columns)
    keep_column_indices = []
    for i, header in enumerate(headers):
        if 'prob.' not in header:
            keep_column_indices.append(i)
    
    # Create the CSV content
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the header row (only for columns we're keeping)
        new_headers = [headers[i] for i in keep_column_indices]
        csv_writer.writerow(new_headers)
        
        # Process each data line
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if not line:  # Skip empty lines
                continue
            
            values = line.split('\t')
            
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
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        # Read the input CSV file
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # Read the header row
            header = next(reader)
            
            # Find the index of the img_name column
            try:
                img_name_index = header.index('img_name')
            except ValueError:
                print("Error: 'img_name' column not found in the CSV file.")
                return False
            
            # Prepare to write the output CSV file
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                
                # Write the header row
                writer.writerow(header)
                
                # Process each row
                rows_processed = 0
                for row in reader:
                    if row and len(row) > img_name_index:
                        # Replace '/' with '_' in the img_name column
                        row[img_name_index] = row[img_name_index].replace('/', '_')
                        writer.writerow(row)
                        rows_processed += 1
        
        print(f"Successfully processed {rows_processed} rows.")
        print(f"Modified CSV saved to '{output_file}'.")
        return True
    
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return False

# Call the function
if __name__ == "__main__":
    
    convert_txt_to_csv('ground_truth.txt', 'ground_truth.csv')
    success = process_csv('ground_truth.csv', 'Emotion6_dataset_annotations.csv')
    
    if success:
        print("CSV processing completed successfully.")
    else:
        print("CSV processing failed.")


import os
from findingemo_light.paper.download_multi import download_data
from findingemo_light.data.read_annotations import read_annotations

# download the findingemo dataset
output_dir = r'D:\dev\VLM-EQ\datasets\findingemo\findingemo_images'
output_csv_path = r'D:\dev\VLM-EQ\datasets\findingemo\findingemo_annotations.csv'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

download_data(target_dir=output_dir)

# read the annotations and save them to a CSV file
ann_data = read_annotations()
ann_data.to_csv(output_csv_path, index=False)

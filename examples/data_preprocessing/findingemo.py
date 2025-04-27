import os

from findingemo_light.data.read_annotations import read_annotations
from findingemo_light.paper.download_multi import download_data

# download the findingemo dataset
output_dir = "/home/pci/yxr/AICA-VLM/datasets/findingemo_images"
output_csv_path = "/home/pci/yxr/AICA-VLM/datasets/findingemo_images/annotations.csv"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

download_data(target_dir=output_dir)

# read the annotations and save them to a CSV file
ann_data = read_annotations()
ann_data.to_csv(output_csv_path, index=False)

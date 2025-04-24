import os

import pandas
from tqdm import tqdm

EmoVIT_annotations = []

reasoning_paths = []
for root, dirs, files in os.walk("./datasets/EmoVIT/reasoning"):
    for file in files:
        reasoning_paths.append(os.path.join(root, file))

n = len(reasoning_paths)

for i in tqdm(range(n), desc="Processing"):
    row = reasoning_paths[i]

    folder_name = os.path.basename(os.path.dirname(row))
    file_name = os.path.splitext(os.path.basename(row))[0]

    with open(row, "r", encoding="utf-8") as f:
        txt_content = f.read().replace("\n", " ")

    new_row = {
        "img_name": file_name,
        "img_folder": f"./datasets/ArtEmis/EmoVIT_images/{folder_name}",
        "emotion_cat": folder_name,
        "emotion_v": "",
        "emotion_a": "",
        "emotion_reasoning": txt_content,
    }
    EmoVIT_annotations.append(new_row)

EmoVIT_annotations = pandas.DataFrame(EmoVIT_annotations)
EmoVIT_annotations.to_csv("./datasets/EmoVIT/EmoVIT_annotations.csv", index=False)

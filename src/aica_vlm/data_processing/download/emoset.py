# EmoSet dataset
# https://vcc.tech/EmoSet


import json
import os
import shutil

import pandas as pd
from tqdm import tqdm


def parse_and_copy_emoset(
    json_paths, image_root_dir, output_image_dir, output_csv_path
):
    os.makedirs(output_image_dir, exist_ok=True)
    data = []

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for entry in tqdm(records, desc=f"Processing {os.path.basename(json_path)}"):
            emotion_cat, img_rel_path, _ = entry

            img_name = os.path.basename(img_rel_path)
            img_folder = os.path.dirname(img_rel_path)
            src_img_path = os.path.join(image_root_dir, img_rel_path)

            if not os.path.exists(src_img_path):
                print(f"[Missing] {src_img_path}")
                continue

            # 拷贝图片
            dst_img_path = os.path.join(output_image_dir, img_name)
            shutil.copy2(src_img_path, dst_img_path)

            data.append(
                {
                    "img_name": img_name,
                    "img_folder": os.path.basename(output_image_dir),
                    "emotion_cat": emotion_cat,
                    "emotion_v": None,
                    "emotion_a": None,
                    "source_dataset": "EmoSet",
                }
            )

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved CSV to {output_csv_path}")
    print(f"Copied {len(df)} images to {output_image_dir}")


if __name__ == "__main__":
    json_dir = r"D:\dev\VLM-EQ\datasets\emoset"
    image_root_dir = r"D:\vlm-eq\EmoSet-118K"
    output_image_dir = r"D:\dev\VLM-EQ\datasets\emoset\emoset_images"
    output_csv_path = r"D:\dev\VLM-EQ\datasets\emoset\emoset_annotations.csv"

    origin_json_dir = r"D:\vlm-eq\EmoSet-118K"

    json_paths = [
        os.path.join(origin_json_dir, "train.json"),
        os.path.join(origin_json_dir, "val.json"),
        os.path.join(origin_json_dir, "test.json"),
    ]

    parse_and_copy_emoset(json_paths, image_root_dir, output_image_dir, output_csv_path)

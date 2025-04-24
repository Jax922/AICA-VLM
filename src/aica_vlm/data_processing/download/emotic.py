# EMOTIC dataset
# https://github.com/rkosti/emotic

import os
import shutil

import pandas as pd
import scipy.io
from tqdm import tqdm


def extract_emotic_annotations_with_copy(
    mat_path, image_root_dir, output_dir, output_csv_path
):
    os.makedirs(output_dir, exist_ok=True)

    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = []

    for split in ["train", "val", "test"]:
        split_data = mat[split]
        for item in tqdm(split_data, desc=f"Processing {split}"):
            img_name = item.filename
            img_folder = item.folder
            persons = item.person

            if not isinstance(persons, (list, tuple)):
                persons = [persons]

            original_img_path = os.path.join(image_root_dir, img_folder, img_name)

            if not os.path.exists(original_img_path):
                print(f"[Missing] {original_img_path}")
                continue

            target_img_path = os.path.join(output_dir, img_name)
            shutil.copy2(original_img_path, target_img_path)

            for person in persons:
                try:
                    cats = person.annotations_categories.categories

                    if isinstance(cats, (list, tuple)):
                        if all(isinstance(c, str) and len(c) == 1 for c in cats):
                            original_labels = ["".join(cats)]
                        else:
                            original_labels = cats
                    elif isinstance(cats, str):
                        original_labels = [cats]
                    elif hasattr(cats, "tolist"):
                        cats_list = cats.tolist()
                        if all(isinstance(c, str) and len(c) == 1 for c in cats_list):
                            original_labels = ["".join(cats_list)]
                        else:
                            original_labels = cats_list
                    else:
                        original_labels = [str(cats)]

                    emotion_cat = "|".join(label.strip() for label in original_labels)

                    emotion_v = person.annotations_continuous.valence
                    emotion_a = person.annotations_continuous.arousal

                    data.append(
                        {
                            "img_name": img_name,
                            "img_folder": os.path.basename(output_dir),
                            "emotion_cat": emotion_cat,
                            "emotion_v": emotion_v,
                            "emotion_a": emotion_a,
                            "source_dataset": "emotic",
                        }
                    )
                except Exception as e:
                    print(f"[Warning] Skipping entry due to error: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"\nSaved cleaned CSV to {output_csv_path}")
    print(f"Total valid entries: {len(df)}")


if __name__ == "__main__":
    mat_path = r"D:\vlm-eq\emotic\Annotations\Annotations\Annotations.mat"
    image_root_dir = r"D:\vlm-eq\emotic\emotic\emotic"
    output_dir = r"D:\dev\VLM-EQ\datasets\EMOTIC\emotic_clean_images"
    output_csv_path = r"D:\dev\VLM-EQ\datasets\EMOTIC\emotic_clean_annotations.csv"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extract_emotic_annotations_with_copy(
        mat_path, image_root_dir, output_dir, output_csv_path
    )

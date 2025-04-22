import os
import json
import shutil
import re
import pandas as pd
from tqdm import tqdm
from dataset import BaseDataset

class ArtemisDataset(BaseDataset):
    def __init__(self, 
        dataset_name: str = 'ArtEmis',
        data_root: str = './datasets/',
        emotion_class: int = 8,
        has_VA: bool = False,
        has_reasoning: bool = True,
    ):
        super().__init__(
            dataset_name = dataset_name,
            data_root = data_root,
            emotion_class = emotion_class,
            has_VA = has_VA,
            has_reasoning = has_reasoning,
        )

    def load_data(self) -> pd.DataFrame:
        artemis_dataset_release_v0 = pd.read_csv('./datasets/ArtEmis/raw_data/artemis_dataset_release_v0.csv')
        ArtEmis_annotations = []
        for i in tqdm(range(len(artemis_dataset_release_v0)), desc='Processing'):
            row = artemis_dataset_release_v0.iloc[i]
            new_row = {
                'img_name': row['painting'],
                'img_folder': f'./datasets/ArtEmis/raw_data/ArtEmis_images/{row["art_style"]}',
                'emotion_cat': row['emotion'],
                'emotion_v': '',
                'emotion_a': '',
                'emotion_reasoning': row['utterance'],
            }
            ArtEmis_annotations.append(new_row)
        ArtEmis_annotations = pd.DataFrame(ArtEmis_annotations)

        return ArtEmis_annotations

    def process_csv(self):

        ArtEmis_annotations = self.load_data()
        ArtEmis_annotations = ArtEmis_annotations[ArtEmis_annotations['emotion_cat'] != 'something else']

        def is_valid_image_name(name):
            return bool(re.match(r'^[a-zA-Z0-9_\-()]+$', name))
        ArtEmis_annotations = ArtEmis_annotations[ArtEmis_annotations['img_name'].apply(is_valid_image_name)]

        ArtEmis_annotations.drop_duplicates(subset=['img_name'], inplace=True)
        
        file_exists_mask = [
            os.path.isfile(os.path.join(row['img_folder'], row['img_name'] + '.jpg'))
            for _, row in ArtEmis_annotations.iterrows()
        ]
        ArtEmis_annotations = ArtEmis_annotations[file_exists_mask]

        return ArtEmis_annotations

    def random_sample(self, nums, random_state: int = 42) -> pd.DataFrame:

        if os.path.exists('./datasets/ArtEmis/ArtEmis_images_sampled'):
            shutil.rmtree('./datasets/ArtEmis/ArtEmis_images_sampled')

        ArtEmis_annotations = self.process_csv()
        ArtEmis_annotations_sampled = ArtEmis_annotations.sample(n=nums, random_state=random_state)

        for _, row in ArtEmis_annotations_sampled.iterrows():
            img_folder = row['img_folder']
            img_name = row['img_name'] + '.jpg'

            src_path = os.path.join(img_folder, img_name)

            sampled_folder = img_folder.replace("raw_data/", "")
            sampled_folder = sampled_folder.replace("ArtEmis_images", 'ArtEmis_images_sampled')
            os.makedirs(sampled_folder, exist_ok=True)
            dst_path = os.path.join(sampled_folder, img_name)

            if os.path.exists(src_path):  # 确保源文件存在
                shutil.copy(src_path, dst_path)
            else:
                raise FileNotFoundError(f"源文件不存在: {src_path}")
            
            row['img_folder'] = sampled_folder

        ArtEmis_annotations_sampled.to_csv('./datasets/ArtEmis/ArtEmis_annotations_sampled.csv', index=False)

        return ArtEmis_annotations_sampled

if __name__ == "__main__":
    artemis_dataset = ArtemisDataset()
    artemis_dataset.random_sample(nums=2500)

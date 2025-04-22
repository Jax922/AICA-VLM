import os
import json
import shutil
import re
import pandas as pd
from tqdm import tqdm
from dataset import BaseDataset

class ArtemisDataset(BaseDataset):
    """ArtEmis dataset processor implementing the BaseDataset interface.
    
    Handles loading, processing, and sampling of the ArtEmis dataset containing
    art images with emotional annotations and reasoning texts.
    """
    
    def __init__(self, 
        dataset_name: str = 'ArtEmis',
        data_root: str = './datasets/',
        emotion_class: int = 8,
        has_VA: bool = False,
        has_reasoning: bool = True,
    ):
        """Initialize ArtEmis dataset processor.
        
        Args:
            dataset_name: Identifier for the dataset. Defaults to 'ArtEmis'.
            data_root: Root directory containing dataset files. 
                Defaults to './datasets/'.
            emotion_class: Number of emotion categories (8 for ArtEmis).
            has_VA: Whether valence-arousal annotations exist. 
                Always False for ArtEmis.
            has_reasoning: Whether emotion reasoning texts exist. 
                Always True for ArtEmis.
        """
        super().__init__(
            dataset_name=dataset_name,
            data_root=data_root,
            emotion_class=emotion_class,
            has_VA=has_VA,
            has_reasoning=has_reasoning,
        )

    def load_data(self) -> pd.DataFrame:
        """Load raw ArtEmis dataset from CSV and transform into standard format.
        
        Returns:
            pd.DataFrame: Processed DataFrame containing:
                - img_name: Painting identifier
                - img_folder: Path to image directory
                - emotion_cat: Emotion category label
                - emotion_v: Valence (empty for ArtEmis)
                - emotion_a: Arousal (empty for ArtEmis) 
                - emotion_reasoning: Textual emotion explanation
        """
        artemis_dataset_release_v0 = pd.read_csv(
            './datasets/ArtEmis/raw_data/artemis_dataset_release_v0.csv'
        )
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
            
        return pd.DataFrame(ArtEmis_annotations)

    def process_csv(self) -> pd.DataFrame:
        """Clean and validate the loaded dataset.
        
        Performs:
        1. Filters out 'something else' emotion category
        2. Validates image filenames
        3. Removes duplicate entries
        4. Verifies image file existence
        
        Returns:
            pd.DataFrame: Cleaned and validated dataset
        """
        ArtEmis_annotations = self.load_data()
        
        # Filter invalid emotion categories
        ArtEmis_annotations = ArtEmis_annotations[
            ArtEmis_annotations['emotion_cat'] != 'something else'
        ]
        
        # Validate filename format
        def is_valid_image_name(name):
            return bool(re.match(r'^[a-zA-Z0-9_\-()]+$', name))
            
        ArtEmis_annotations = ArtEmis_annotations[
            ArtEmis_annotations['img_name'].apply(is_valid_image_name)
        ]
        
        # Remove duplicates
        ArtEmis_annotations.drop_duplicates(subset=['img_name'], inplace=True)
        
        # Verify image files exist
        file_exists_mask = [
            os.path.isfile(os.path.join(row['img_folder'], row['img_name'] + '.jpg'))
            for _, row in ArtEmis_annotations.iterrows()
        ]
        
        return ArtEmis_annotations[file_exists_mask]

    def random_sample(self, nums: int, random_state: int = 42) -> pd.DataFrame:
        """Create a randomized subset of the dataset.
        
        Args:
            nums: Number of samples to select
            random_state: Random seed for reproducibility. Defaults to 42.
            
        Returns:
            pd.DataFrame: Sampled subset of the dataset
            
        Raises:
            FileNotFoundError: If source image files are missing
        """
        # Clear existing samples
        if os.path.exists('./datasets/ArtEmis/ArtEmis_images_sampled'):
            shutil.rmtree('./datasets/ArtEmis/ArtEmis_images_sampled')

        ArtEmis_annotations = self.process_csv()
        ArtEmis_annotations_sampled = ArtEmis_annotations.sample(
            n=nums, 
            random_state=random_state
        )

        # Copy sampled images
        for _, row in ArtEmis_annotations_sampled.iterrows():
            img_folder = row['img_folder']
            img_name = row['img_name'] + '.jpg'
            src_path = os.path.join(img_folder, img_name)

            # Prepare destination
            sampled_folder = img_folder.replace("raw_data/", "")
            sampled_folder = sampled_folder.replace(
                "ArtEmis_images", 
                'ArtEmis_images_sampled'
            )
            os.makedirs(sampled_folder, exist_ok=True)
            dst_path = os.path.join(sampled_folder, img_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                raise FileNotFoundError(f"Source file missing: {src_path}")
            
            row['img_folder'] = sampled_folder

        # Save metadata
        ArtEmis_annotations_sampled.to_csv(
            './datasets/ArtEmis/ArtEmis_annotations_sampled.csv', 
            index=False
        )

        return ArtEmis_annotations_sampled

if __name__ == "__main__":
    artemis_dataset = ArtemisDataset()
    artemis_dataset.random_sample(nums=2500)
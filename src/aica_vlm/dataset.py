import os
import shutil
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class BaseDataset(ABC):
    """Abstract base class for dataset processing pipelines.

    Provides common interface for data loading, processing and sampling operations.
    All concrete dataset classes should inherit from this base class.
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        emotion_class: int,
        has_VA: bool,
        has_reasoning: bool,
    ):
        """Initialize the dataset processor with essential configurations.

        Args:
            dataset_name: Name identifier for the dataset (e.g., 'ArtEmis')
            data_root: Root directory path where dataset files are stored
            emotion_class: Number of emotion classification categories
            has_VA: Whether the dataset contains valence-arousal annotations
            has_reasoning: Whether the dataset contains emotion reasoning texts
        """
        self.dataset_name = dataset_name
        self.data_root = data_root

        self.label_config = {
            "emotion_class": emotion_class,
            "has_VA": has_VA,
            "has_reasoning": has_reasoning,
        }

    @abstractmethod
    def load_data(self):
        """Load raw dataset from source files.

        Returns:
            pd.DataFrame: Loaded raw data in standardized DataFrame format
        """
        pass

    @abstractmethod
    def process_csv(self):
        """Perform data cleaning and standardization.

        Typical operations include:
        - Handling missing values
        - Format normalization
        - Feature engineering

        Returns:
            pd.DataFrame: Processed data in standardized format
        """
        pass

    @abstractmethod
    def random_sample(self, nums: int, random_state: int = 42):
        """Create randomly sampled subset of the dataset.

        Args:
            nums: Number of samples to select
            random_state: Seed for reproducible sampling

        Returns:
            pd.DataFrame: Sampled subset of data
        """
        pass


class GenericDataset(BaseDataset):
    def load_data(self):
        csv_path = os.path.join(self.data_root, "annotations.csv")
        return pd.read_csv(csv_path)

    def process_csv(self):
        df = self.load_data()
        df = df.dropna(subset=["img_name"])
        df["img_path"] = df.apply(
            lambda x: os.path.join(self.data_root, "images", x["img_name"]), axis=1
        )
        return df

    def random_sample(self, nums, random_state=42):
        df = self.process_csv()
        return df.sample(n=nums, random_state=random_state).reset_index(drop=True)


def build_random_benchmark_dataset(
    dataset_cfgs: list,
    total_num: Union[int, str],
    output_dir: str,
    random_state: int = 42,
):
    print(f"Building benchmark dataset with unified random sampling")
    print(f"Target: {total_num} images")

    all_data = []

    for cfg in dataset_cfgs:
        dataset = GenericDataset(
            dataset_name=cfg["name"],
            data_root=cfg["path"],
            emotion_class=cfg["emotion_class"],
            has_VA=cfg["has_VA"],
            has_reasoning=cfg["has_reasoning"],
        )
        df = dataset.process_csv()
        df["source_dataset"] = cfg["name"]
        all_data.append(df)

    merged_df = (
        pd.concat(all_data)
        .dropna(subset=["img_path"])
        .drop_duplicates(subset=["img_path"])
        .reset_index(drop=True)
    )
    print(f"Total unique images after merge: {len(merged_df)}")

    is_all = isinstance(total_num, str) and total_num.lower() == "all"

    if not is_all:
        total_num = int(total_num)
        if len(merged_df) < total_num:
            raise ValueError(
                f"Only {len(merged_df)} unique images available, cannot satisfy total_num={total_num}"
            )
        sampled_df = merged_df.sample(
            n=total_num, random_state=random_state
        ).reset_index(drop=True)
    else:
        sampled_df = merged_df
        total_num = len(sampled_df)
        print(f"Using all {total_num} images without sampling.")

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    for i, row in sampled_df.iterrows():
        src = row["img_path"]
        dst = os.path.join(output_dir, "images", f"{i:05d}.jpg")
        shutil.copy(src, dst)
        sampled_df.at[i, "img_name"] = f"{i:05d}.jpg"

    sampled_df.drop(columns=["img_path"], inplace=True)
    sampled_df.to_csv(os.path.join(output_dir, "annotations.csv"), index=False)

    print(f"✅ Done! {total_num} samples saved to {output_dir}")


def build_balanced_benchmark_dataset(
    dataset_cfgs, total_num, output_dir, random_state=42
):
    print(f"Building balanced benchmark dataset...")
    all_data = []

    for cfg in dataset_cfgs:
        dataset = GenericDataset(
            dataset_name=cfg["name"],
            data_root=cfg["path"],
            emotion_class=cfg["emotion_class"],
            has_VA=cfg["has_VA"],
            has_reasoning=cfg["has_reasoning"],
        )
        df = dataset.process_csv()
        df["source_dataset"] = cfg["name"]
        all_data.append(df)

    merged_df = (
        pd.concat(all_data)
        .dropna(subset=["emotion_cat", "img_path"])
        .drop_duplicates(subset=["img_path"])
        .reset_index(drop=True)
    )

    print(f"Total merged samples: {len(merged_df)}")

    emotion_counts = merged_df["emotion_cat"].value_counts()
    emotion_labels = emotion_counts.index.tolist()
    num_classes = len(emotion_labels)

    num_per_class = total_num // num_classes
    print(
        f"Sampling {num_per_class} per class × {num_classes} classes = {num_per_class * num_classes}"
    )

    balanced_samples = []
    for label in emotion_labels:
        class_subset = merged_df[merged_df["emotion_cat"] == label]
        if len(class_subset) < num_per_class:
            print(
                f"⚠️ Class {label} only has {len(class_subset)} samples, not enough for {num_per_class}"
            )
        sampled = class_subset.sample(
            n=min(num_per_class, len(class_subset)), random_state=random_state
        )
        balanced_samples.append(sampled)

    final_df = (
        pd.concat(balanced_samples)
        .drop_duplicates(subset=["img_path"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    for i, row in final_df.iterrows():
        src = row["img_path"]
        dst = os.path.join(output_dir, "images", f"{i:05d}.jpg")
        shutil.copy(src, dst)
        final_df.at[i, "img_name"] = f"{i:05d}.jpg"

    final_df.drop(columns=["img_path"], inplace=True)
    final_df.to_csv(os.path.join(output_dir, "annotations.csv"), index=False)

    print(f"Balanced benchmark saved: {len(final_df)} images in {output_dir}")


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import yaml

    with open(r"D:\dev\VLM-EQ\benchmark_datasets\EU_observer_emotion.yaml") as f:
        config = yaml.safe_load(f)

    print("Building benchmark {} dataset.".format(config["task_name"]))
    build_random_benchmark_dataset(
        dataset_cfgs=config["datasets"],
        total_num=config["total_num"],
        output_dir=config["output_dir"],
    )

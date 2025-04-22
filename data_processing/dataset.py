from abc import ABC, abstractmethod

class BaseDataset(ABC):
    def __init__(
        self,
        dataset_name: str,
        data_root: str,
        emotion_class: int,
        has_VA: bool,
        has_reasoning: bool,
    ):
        self.dataset_name = dataset_name
        self.data_root = data_root

        self.label_config = {
            "emotion_class": emotion_class,
            "has_VA": has_VA, 
            "has_reasoning": has_reasoning
        }

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def process_csv(self):
        pass

    @abstractmethod
    def random_sample(self, nums, random_state: int = 42):
        pass

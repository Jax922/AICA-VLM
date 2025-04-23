from abc import ABC, abstractmethod

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
            "has_reasoning": has_reasoning
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
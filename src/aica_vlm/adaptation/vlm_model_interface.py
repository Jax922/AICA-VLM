from abc import ABC, abstractmethod

class VLMModelInterface(ABC):
    """Abstract base class defining a unified VLM model interface"""

    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass

    @abstractmethod
    def preprocess_prompt(self, prompt: dict):
        """Convert the unified prompt into the model's input format"""
        pass

    @abstractmethod
    def infer(self, inputs):
        """Perform inference"""
        pass
    
    @abstractmethod
    def predict_with_prompt(self, prompt: dict):
        """Perform inference using the unified prompt"""
        pass
    
class VLMModelFactory(ABC):
    """Abstract factory class for creating different VLM model instances"""

    @abstractmethod
    def create_model(self) -> VLMModelInterface:
        """Create a model instance"""
        pass

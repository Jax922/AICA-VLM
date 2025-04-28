from abc import ABC, abstractmethod

class VLMModelInterface(ABC):
    """Abstract base class defining a unified VLM model interface"""

    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass

    @abstractmethod
    def process_instruction(self, instruction: dict):
        """Convert the unified prompt into the model's input format"""
        pass

    @abstractmethod
    def inference(self, instruction: dict):
        """Perform inference"""
        pass
    
class VLMModelFactory(ABC):
    """Abstract factory class for creating different VLM model instances"""

    @abstractmethod
    def create_model(self) -> VLMModelInterface:
        """Create a model instance"""
        pass

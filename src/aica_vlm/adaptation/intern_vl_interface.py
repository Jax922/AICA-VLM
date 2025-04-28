import sys
import json
import torch
from typing import List, Dict, Tuple, Optional
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T
from aica_vlm.adaptation.vlm_model_interface import VLMModelInterface, VLMModelFactory


# Image normalization constants for ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(input_size: int) -> T.Compose:
    """
    Construct preprocessing pipeline for images.
    
    Args:
        input_size: Target size for resizing (width and height)
        
    Returns:
        Composition of torchvision transforms for image preprocessing
    """
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def dynamic_image_split(
    image: Image.Image,
    min_blocks: int = 1,
    max_blocks: int = 12,
    block_size: int = 448,
    include_thumbnail: bool = False
) -> List[Image.Image]:
    """
    Dynamically split image into optimally sized blocks based on aspect ratio.
    
    Args:
        image: Input PIL Image object
        min_blocks: Minimum number of blocks to create
        max_blocks: Maximum number of blocks to create
        block_size: Target size for each block
        include_thumbnail: Whether to append a thumbnail of the original image
        
    Returns:
        List of processed image blocks
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate all possible block configurations within constraints
    possible_configs = sorted(
        {(w, h) for n in range(min_blocks, max_blocks + 1) 
         for w in range(1, n + 1) for h in range(1, n + 1) 
         if w * h <= max_blocks and w * h >= min_blocks},
        key=lambda x: x[0] * x[1]
    )

    # Find configuration with closest aspect ratio match
    best_config = min(
        possible_configs,
        key=lambda ratio: abs(aspect_ratio - (ratio[0] / ratio[1]))
    )

    # Resize and split image
    target_width = block_size * best_config[0]
    target_height = block_size * best_config[1]
    resized_img = image.resize((target_width, target_height))

    # Generate blocks
    blocks = []
    for i in range(best_config[0] * best_config[1]):
        box = (
            (i % best_config[0]) * block_size,
            (i // best_config[0]) * block_size,
            ((i % best_config[0]) + 1) * block_size,
            ((i // best_config[0]) + 1) * block_size
        )
        blocks.append(resized_img.crop(box))

    # Optionally add thumbnail
    if include_thumbnail and len(blocks) > 1:
        blocks.append(image.resize((block_size, block_size)))

    return blocks


def preprocess_image(
    image_path: str, 
    input_size: int = 448, 
    max_blocks: int = 12
) -> torch.Tensor:
    """
    Load and preprocess image into model-ready tensor format.
    
    Args:
        image_path: Path to image file
        input_size: Target size for processing
        max_blocks: Maximum number of blocks to split into
        
    Returns:
        Tensor containing preprocessed image data
    """
    image = Image.open(image_path).convert('RGB')
    transform = build_image_transform(input_size)
    image_blocks = dynamic_image_split(
        image,
        block_size=input_size,
        include_thumbnail=True,
        max_blocks=max_blocks
    )
    return torch.stack([transform(img) for img in image_blocks])


class InternVL(VLMModelInterface):
    """Implementation of VLMModelInterface for InternVL vision-language models."""
    
    def __init__(self, model_name: str):
        """
        Initialize InternVL model instance.
        
        Args:
            model_name: Name/path of the model to load
        """
        self.model_name = model_name
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer = None

    def load_model(self) -> None:
        """
        Load model weights and initialize tokenizer.
        
        Raises:
            ValueError: If model name is not recognized
            ImportError: If required dependencies are missing
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            
            if "InternVL2_5" in self.model_name or "InternVL3" in self.model_name:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True
                ).eval().cuda()
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=False
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
        except ImportError as e:
            raise ImportError(f"Failed to load required dependencies: {str(e)}")

    def process_instruction(self, instruction: Dict) -> Tuple[str, torch.Tensor]:
        """
        Convert unified instruction format to model inputs.
        
        Args:
            instruction: Dictionary containing:
                - messages: List of message dicts
                - images: List of image paths
                
        Returns:
            Tuple containing:
                - Text prompt/question
                - Preprocessed image tensor
                
        Raises:
            ValueError: If input format is invalid
        """
        try:
            question = instruction["messages"][0]["content"]
            img_path = instruction["images"][0]
            
            if not isinstance(question, str) or not isinstance(img_path, str):
                raise ValueError("Input validation failed: content must be string")
                
            pixel_values = preprocess_image(img_path, max_blocks=12)
            return question, pixel_values.to(torch.bfloat16).cuda()
            
        except (KeyError, IndexError) as e:
            raise ValueError(f"Malformed instruction format: {str(e)}")

    def inference(self, instruction: Dict) -> str:
        """
        Perform single-instruction inference.
        
        Args:
            instruction: Input instruction dictionary
            
        Returns:
            Generated text response
        """
        question, pixel_values = self.process_instruction(instruction)
        generation_config = {
            'max_new_tokens': 256,
            'do_sample': True
        }
        return self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config
        )

    def batch_inference(self, instructions: List[Dict]) -> List[str]:
        """
        Perform batched inference on multiple instructions.
        
        Args:
            instructions: List of instruction dictionaries
            
        Returns:
            List of generated responses
        """
        questions = []
        pixel_values_list = []
        
        for instruction in instructions:
            question, pixel_values = self.process_instruction(instruction)
            questions.append(question)
            pixel_values_list.append(pixel_values)
            
        concatenated_pixels = torch.cat(pixel_values_list, dim=0)
        
        generation_config = {
            'max_new_tokens': 256,
            'do_sample': True
        }
        
        return self.model.batch_chat(
            self.tokenizer,
            concatenated_pixels,
            num_patches_list=[pv.size(0) for pv in pixel_values_list],
            questions=questions,
            generation_config=generation_config
        )


class InternVLFactory(VLMModelFactory):
    """Factory class for creating InternVL model instances."""
    
    def __init__(self, model_name: str):
        """
        Initialize factory with model specification.
        
        Args:
            model_name: Name/path of the model to create
        """
        self.model_name = model_name

    def create_model(self) -> VLMModelInterface:
        """
        Instantiate and initialize an InternVL model.
        
        Returns:
            Initialized InternVL model instance
        """
        model = InternVL(self.model_name)
        model.load_model()
        return model


if __name__ == '__main__':
    # Example usage with error handling
    try:
        with open('datasets/abstract/instruction.json', 'r', encoding='utf-8') as f:
            instructions = json.load(f)

        model_name = "models/OpenGVLab/InternVL3-8B-Instruct"
        
        factory = InternVLFactory(model_name)
        model = factory.create_model()
        
        for instruction in instructions:
            try:
                result = model.inference(instruction)
                print(result)
            except Exception as e:
                print(f"Error processing instruction: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
import json
import sys
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from aica_vlm.adaptation.vlm_model_interface import VLMModelFactory, VLMModelInterface

def resize_image(image_path: str, max_width: int, max_height: int) -> Image.Image:
    """
    Resize an image if its width or height exceeds specified maximum dimensions.

    Args:
        image_path: Path to the image file.
        max_width: Maximum allowable width.
        max_height: Maximum allowable height.

    Returns:
        Resized PIL Image object.
    """
    image = Image.open(image_path)
    original_width, original_height = image.size

    if original_width > max_width or original_height > max_height:
        scale_factor = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image

class Llava(VLMModelInterface):
    """Implementation of VLMModelInterface for LLaVA vision-language models."""

    def __init__(self, model_type: str, model_path: str):
        """
        Initialize LLaVA model instance.
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_class: Optional[type] = None
        self.processor_class: Optional[type] = None
        self.model = None
        self.processor = None
        self.max_width = 1024  # Define maximum width threshold
        self.max_height = 1024  # Define maximum height threshold
        
    def load_model(self) -> None:
        if self.model_type == "LLaVA-onevision":
            from transformers import (
                AutoProcessor,
                LlavaOnevisionForConditionalGeneration,
            )

            self.model_class = LlavaOnevisionForConditionalGeneration
            self.processor_class = AutoProcessor
        elif self.model_type == "LLaVA-1.6":
            from transformers import (
                LlavaNextForConditionalGeneration,
                LlavaNextProcessor,
            )

            self.model_class = LlavaNextForConditionalGeneration
            self.processor_class = LlavaNextProcessor
        else:
            raise ValueError(f"Unsupported LLaVA model: {self.model_path}")

        self.model = self.model_class.from_pretrained(
            self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to("cuda")

        self.processor = self.processor_class.from_pretrained(self.model_path)

    def process_instruction(self, instruction: Dict) -> Tuple[str, Image.Image]:
        try:
            user_content = instruction["messages"][0]["content"]
            img_path = instruction["images"][0]

            if not isinstance(user_content, str) or not isinstance(img_path, str):
                raise ValueError(
                    "Input validation failed: content and image path must be strings"
                )

            prompt_text = user_content.split("<image>", 1)[1].strip()

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )

            image = resize_image(img_path, self.max_width, self.max_height)
            
            return prompt, image

        except (KeyError, IndexError) as e:
            raise ValueError(f"Malformed instruction format: {str(e)}")
        except FileNotFoundError as e:
            raise ValueError(f"Image file not found: {str(e)}")

    def inference(self, instruction: Dict) -> str:
        
        prompt, image = self.process_instruction(instruction)

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            0, torch.float16
        )

        output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        output_text = self.processor.decode(output[0], skip_special_tokens=True)

        marker = "[/INST]"
        marker_index = output_text.find(marker)
        
        if marker_index != -1:
            output_text =  output_text[marker_index + len(marker):].strip()

        marker = "assistant\n"
        marker_index = output_text.find(marker)
        
        if marker_index != -1:
            output_text =  output_text[marker_index + len(marker):].strip()

        marker = "ASSISTANT: "
        marker_index = output_text.find(marker)
        
        if marker_index != -1:
            output_text =  output_text[marker_index + len(marker):].strip()
            
        return output_text

    def batch_inference(self, instructions: list[dict]):
        # TODO
        pass
    
class LlavaFactory(VLMModelFactory):
    """Factory class for creating LLaVA model instances."""

    def __init__(self, model_type: str, model_path: str):
        """
        Initialize factory with model specification.
        """
        self.model_type = model_type
        self.model_path = model_path

    def create_model(self) -> VLMModelInterface:
        """
        Create and initialize a LLaVA model instance.

        Returns:
            Initialized LLaVA model instance
        """
        model = Llava(self.model_type, self.model_path)
        model.load_model()
        return model

import sys
import json
from typing import List, Dict, Optional
from qwen_vl_utils import process_vision_info
from aica_vlm.adaptation.vlm_model_interface import VLMModelInterface, VLMModelFactory


class QwenVL(VLMModelInterface):
    """Implementation of VLMModelInterface for Qwen Vision-Language models."""
    
    def __init__(self, model_name: str):
        """
        Initialize QwenVL model instance.
        
        Args:
            model_name: Name of the model to load (e.g., "Qwen/Qwen2-VL-7B-Instruct")
        """
        self.model_name = model_name
        self.model_class: Optional[type] = None
        self.processor_class: Optional[type] = None
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """
        Dynamically load model and processor based on model name.
        
        Raises:
            ValueError: If the model name is not recognized.
            ImportError: If required dependencies are missing.
        """
        try:
            if "Qwen2.5-VL" in self.model_name:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                self.model_class = Qwen2_5_VLForConditionalGeneration
            elif "Qwen2-VL" in self.model_name:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                self.model_class = Qwen2VLForConditionalGeneration
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            self.processor_class = AutoProcessor
            self.model = self.model_class.from_pretrained(
                self.model_name, 
                torch_dtype="auto", 
                device_map="auto"
            )
            self.processor = self.processor_class.from_pretrained(self.model_name)
            
        except ImportError as e:
            raise ImportError(f"Failed to import required modules: {str(e)}")

    def process_instruction(self, instruction: Dict) -> List[Dict]:
        """
        Convert unified instruction format to model-specific input format.
        
        Args:
            instruction: Dictionary containing:
                - messages: List of message dictionaries
                - images: List of image paths
                
        Returns:
            List of message dictionaries in QwenVL format
            
        Raises:
            ValueError: If input format is invalid
        """
        try:
            user_content = instruction["messages"][0]["content"]
            img_path = instruction["images"][0]
            
            if not isinstance(user_content, str) or not isinstance(img_path, str):
                raise ValueError("Invalid input format: content and image path must be strings")
                
            prompt_text = user_content.split("<image>", 1)[1].strip()
            
            return [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt_text}
                ]
            }]
            
        except (KeyError, IndexError) as e:
            raise ValueError(f"Malformed instruction: {str(e)}")

    def inference(self, instruction: Dict) -> List[str]:
        """
        Perform single-instruction inference.
        
        Args:
            instruction: Input instruction dictionary
            
        Returns:
            List of generated text outputs
        """
        message = self.process_instruction(instruction)
        
        # Prepare inputs
        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(message)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        
        return self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

    def batch_inference(self, instructions: List[Dict]) -> List[str]:
        """
        Perform batched inference on multiple instructions.
        
        Args:
            instructions: List of instruction dictionaries
            
        Returns:
            List of generated text outputs
        """
        messages = [self.process_instruction(inst) for inst in instructions]
        
        # Prepare batch inputs
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) 
            for msg in messages
        ]
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # Generate outputs
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )


class QwenVLFactory(VLMModelFactory):
    """Factory class for creating QwenVL model instances."""
    
    def __init__(self, model_name: str):
        """
        Initialize factory with model name.
        
        Args:
            model_name: Name of the model to create
        """
        self.model_name = model_name

    def create_model(self) -> VLMModelInterface:
        """
        Create and initialize a QwenVL model instance.
        
        Returns:
            Initialized QwenVL model instance
        """
        model = QwenVL(self.model_name)
        model.load_model()
        return model


if __name__ == '__main__':
    # Example usage
    with open('datasets/abstract/instruction.json', 'r', encoding='utf-8') as f:
        instructions = json.load(f)

    model_name = 'Qwen/Qwen2.5-VL-3B-Instruct'
    factory = QwenVLFactory(model_name)
    model = factory.create_model()
    
    for instruction in instructions:
        try:
            result = model.inference(instruction)
            print(result)
        except Exception as e:
            print(f"Error processing instruction: {str(e)}")
            continue
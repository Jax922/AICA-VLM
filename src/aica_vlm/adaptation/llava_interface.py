import json
import sys
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from aica_vlm.adaptation.vlm_model_interface import VLMModelFactory, VLMModelInterface


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

            return prompt, Image.open(img_path)

        except (KeyError, IndexError) as e:
            raise ValueError(f"Malformed instruction format: {str(e)}")
        except FileNotFoundError as e:
            raise ValueError(f"Image file not found: {str(e)}")

    def inference(self, instruction: Dict) -> str:
        
        prompt, raw_image = self.process_instruction(instruction)

        inputs = self.processor(images=raw_image, text=prompt, return_tensors="pt").to(
            0, torch.float16
        )

        output = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
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
        
    def batch_inference(self, instructions: List[Dict]) -> List[str]:
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


if __name__ == "__main__":

    model_type = 'LLaVA-1.6'
    model_path = 'models/llava-hf/llava-v1.6-mistral-7b-hf'

    from aica_vlm.adaptation.instruction_load import InstructionLoader
    
    instruction_loader  = InstructionLoader("/public/home/202320163218/yxr_code/LLM_Workspace/AICA-VLM/datasets/benchmark/abstract/instruction.json",
                                            "/public/home/202320163218/yxr_code/LLM_Workspace/AICA-VLM/datasets/benchmark/abstract/images")
    
    instruction_loader.load()

    try:
        factory = LlavaFactory(model_type, model_path)
        model = factory.create_model()

        instructions = instruction_loader.get_instructions()
        
        for instruction in instructions:
            try:
                result = model.inference(instruction)
                print(result)
            except Exception as e:
                print(f"Error processing instruction: {str(e)}")
                continue

    except Exception as e:
        print(f"Failed to initialize model: {str(e)}")
        sys.exit(1)

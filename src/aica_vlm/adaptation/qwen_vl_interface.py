import json
import sys
from typing import Dict, List, Optional

from qwen_vl_utils import process_vision_info

from aica_vlm.adaptation.vlm_model_interface import VLMModelFactory, VLMModelInterface


class QwenVL(VLMModelInterface):
    """Implementation of VLMModelInterface for Qwen Vision-Language models."""

    def __init__(self, model_type: str, model_path: str):
        """
        Initialize QwenVL model instance.
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_class = None
        self.processor_class = None
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """
        Dynamically load model and processor based on model name.

        Raises:
            ValueError: If the model name is not recognized.
            ImportError: If required dependencies are missing.
        """
        if "Qwen2.5-VL" == self.model_type:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self.model_class = Qwen2_5_VLForConditionalGeneration
            self.processor_class = AutoProcessor
        elif "Qwen2-VL" in self.model_type:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            self.model_class = Qwen2VLForConditionalGeneration
            self.processor_class = AutoProcessor
        else:
            raise ValueError(f"Unrecognized model name: {self.model_type}")

        # Load the model and processor
        self.model = self.model_class.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = self.processor_class.from_pretrained(self.model_path)

    def process_instruction(self, instruction: dict) -> list:
        user_content = instruction["messages"][0]["content"]
        img_path = instruction["images"][0]

        # Ensure the image path and content are valid
        if not isinstance(user_content, str) or not isinstance(img_path, str):
            raise ValueError(
                "Invalid prompt format: 'messages' or 'images' is not a string."
            )

        # Extract the prompt text
        full_prompt = user_content.split("<image>", 1)[1].strip()
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        return message

    def inference(self, instruction: dict):
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
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text

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
        generated_ids = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
        ]

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


class QwenVLFactory(VLMModelFactory):
    """Factory class for creating QwenVL model instances."""

    def __init__(self, model_type: str, model_path: str):
        """
        Initialize factory with model name.
        """
        self.model_type = model_type
        self.model_path = model_path

    def create_model(self) -> VLMModelInterface:
        """
        Create and initialize a QwenVL model instance.

        Returns:
            Initialized QwenVL model instance
        """
        model = QwenVL(self.model_type, self.model_path)
        model.load_model()
        return model


if __name__ == "__main__":
    # Example usage
    with open("datasets/abstract/instruction.json", "r", encoding="utf-8") as f:
        instructions = json.load(f)

    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    factory = QwenVLFactory(model_name)
    model = factory.create_model()

    for instruction in instructions:
        try:
            result = model.inference(instruction)
            print(result)
        except Exception as e:
            print(f"Error processing instruction: {str(e)}")
            continue

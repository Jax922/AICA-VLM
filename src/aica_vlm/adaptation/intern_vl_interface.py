import json
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from aica_vlm.adaptation.vlm_model_interface import VLMModelFactory, VLMModelInterface

# Image normalization constants for ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def dynamic_image_split(
    image: Image.Image,
    min_blocks: int = 1,
    max_blocks: int = 12,
    block_size: int = 448,
    include_thumbnail: bool = False,
) -> List[Image.Image]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate all possible block configurations within constraints
    possible_configs = sorted(
        {
            (w, h)
            for n in range(min_blocks, max_blocks + 1)
            for w in range(1, n + 1)
            for h in range(1, n + 1)
            if w * h <= max_blocks and w * h >= min_blocks
        },
        key=lambda x: x[0] * x[1],
    )

    # Find configuration with closest aspect ratio match
    best_config = min(
        possible_configs, key=lambda ratio: abs(aspect_ratio - (ratio[0] / ratio[1]))
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
            ((i // best_config[0]) + 1) * block_size,
        )
        blocks.append(resized_img.crop(box))

    # Optionally add thumbnail
    if include_thumbnail and len(blocks) > 1:
        blocks.append(image.resize((block_size, block_size)))

    return blocks


def preprocess_image(
    image_path: str, input_size: int = 448, max_blocks: int = 12
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = build_image_transform(input_size)
    image_blocks = dynamic_image_split(
        image, block_size=input_size, include_thumbnail=True, max_blocks=max_blocks
    )
    return torch.stack([transform(img) for img in image_blocks])


class InternVL(VLMModelInterface):
    """Implementation of VLMModelInterface for InternVL vision-language models."""

    def __init__(self, model_type: str, model_path: str):
        self.model_type = model_type
        self.model_path = model_path
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer = None

    def load_model(self) -> None:
        if self.model_type in ["InternVL2_5", "InternVL3"]:
            self.model = (
                AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=True,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, use_fast=False
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_path}")

    def process_instruction(self, instruction: Dict) -> Tuple[str, torch.Tensor]:
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
        question, pixel_values = self.process_instruction(instruction)
        generation_config = {"max_new_tokens": 512, "do_sample": True}
        return self.model.chat(
            self.tokenizer, pixel_values, question, generation_config
        )

    def batch_inference(self, instructions: List[Dict]) -> List[str]:
        questions = []
        pixel_values_list = []

        for instruction in instructions:
            question, pixel_values = self.process_instruction(instruction)
            questions.append(question)
            pixel_values_list.append(pixel_values)

        concatenated_pixels = torch.cat(pixel_values_list, dim=0)

        generation_config = {"max_new_tokens": 512, "do_sample": True}

        return self.model.batch_chat(
            self.tokenizer,
            concatenated_pixels,
            num_patches_list=[pv.size(0) for pv in pixel_values_list],
            questions=questions,
            generation_config=generation_config,
        )


class InternVLFactory(VLMModelFactory):
    """Factory class for creating InternVL model instances."""

    def __init__(self, model_type: str, model_path: str):
        """
        Initialize factory with model specification.
        """
        self.model_type = model_type
        self.model_path = model_path

    def create_model(self) -> VLMModelInterface:
        """
        Instantiate and initialize an InternVL model.

        Returns:
            Initialized InternVL model instance
        """
        model = InternVL(self.model_type, self.model_path)
        model.load_model()
        return model


if __name__ == "__main__":
    # Example usage with error handling
    try:
        with open("datasets/abstract/instruction.json", "r", encoding="utf-8") as f:
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

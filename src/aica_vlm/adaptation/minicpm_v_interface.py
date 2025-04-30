import sys

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from aica_vlm.adaptation.vlm_model_interface import VLMModelFactory, VLMModelInterface


class MiniCPMV(VLMModelInterface):
    def __init__(self, model_type: str, model_path: str):
        """
        Initialize MiniCPM-V model.

        Args:
            model_name (str): Model name, e.g., "openbmb/MiniCPM-V-2_6".
        """
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Dynamically load the model and tokenizer based on the model name.
        """
        if self.model_type in ["MiniCPM-V", "MiniCPM-o"]:
            self.model = (
                AutoModel.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    attn_implementation="sdpa",  # Use SDPA attention
                    torch_dtype=torch.bfloat16,
                )
                .eval()
                .cuda()
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        else:
            raise ValueError(f"Unrecognized model name: {self.model_path}")

    def process_instruction(self, instruction: dict):
        user_content = instruction["messages"][0]["content"]
        img_path = instruction["images"][0]

        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")

        # Prepare the message
        message = [{"role": "user", "content": [image, user_content]}]

        return message

    def inference(self, instruction: dict):
        message = self.process_instruction(instruction)

        output_text = self.model.chat(
            image=None, msgs=message, tokenizer=self.tokenizer
        )

        return output_text

    def batch_inference(self, instructions: list[dict]):
        # TODO
        pass


class MiniCPMVFactory(VLMModelFactory):
    def __init__(self, model_type: str, model_path: str):
        """
        Initialize MiniCPM-V factory.
        """
        self.model_type = model_type
        self.model_path = model_path

    def create_model(self) -> VLMModelInterface:
        """
        Create a specific version of the MiniCPM-V model instance.

        Returns:
            VLMModelInterface: An instance of the MiniCPM-V model.
        """
        model = MiniCPMV(self.model_type, self.model_path)
        model.load_model()
        return model


if __name__ == "__main__":
    import json

    with open("./datasets/emoset/instruction.json", "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # Specify the model name
    model_name = ".models/openbmb/MiniCPM-V-2_6"

    # Create the model using the factory
    minicpm_factory = MiniCPMVFactory(model_name)
    minicpm_model = minicpm_factory.create_model()

    for instruction in instructions:
        try:
            result = minicpm_model.inference(instruction)
        except Exception as e:
            continue
        print(result)

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

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

    image = image.convert("RGB")
    
    return image

class MiniCPM(VLMModelInterface):
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
        self.max_width = 1024  # Define maximum width threshold
        self.max_height = 1024  # Define maximum height threshold
        
    def load_model(self):
        """
        Dynamically load the model and tokenizer based on the model name.
        """
        if self.model_type in ["MiniCPM-V"]:
            self.model = AutoModel.from_pretrained(self.model_path, 
                                                   trust_remote_code=True, 
                                                   attn_implementation='flash_attention_2', 
                                                   torch_dtype=torch.bfloat16).eval().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        elif self.model_type in ["MiniCPM-o"]:
            self.model = AutoModel.from_pretrained(self.model_path, 
                                                   trust_remote_code=True,
                                                   attn_implementation='flash_attention_2',
                                                   torch_dtype=torch.bfloat16,
                                                   init_vision=True,
                                                   init_audio=False,
                                                   init_tts=False).eval().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            raise ValueError(f"Unrecognized model name: {self.model_path}")

    def process_instruction(self, instruction: dict):
        user_content = instruction["messages"][0]["content"]
        img_path = instruction["images"][0]

        # Resize image if needed
        image = resize_image(img_path, self.max_width, self.max_height)

        user_content = user_content.split("<image>", 1)[1].strip()
        # Prepare the message
        message = [
            {
                "role": "user", 
                "content": [image, user_content]
            }
        ]

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


class MiniCPMFactory(VLMModelFactory):
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
        model = MiniCPM(self.model_type, self.model_path)
        model.load_model()
        return model


if __name__ == "__main__":
    
    import json

    with open("datasets/benchmark/emoset/instruction.json", "r", encoding="utf-8") as f:
        instructions = json.load(f)

    # Specify the model name
    model_type = "MiniCPM-V"
    model_path = "models/openbmb/MiniCPM-V-2_6"

    # Create the model using the factory
    minicpm_factory = MiniCPMFactory(model_type, model_path)
    minicpm_model = minicpm_factory.create_model()

    for instruction in instructions:
        result = minicpm_model.inference(instruction)
        print(result)

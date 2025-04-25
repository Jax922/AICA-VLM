import torch
from PIL import Image
from src.aica_vlm.adaptation.vlm_model_interface import VLMModelInterface, VLMModelFactory

class MiniCPMV(VLMModelInterface):
    def __init__(self, model_name: str):
        """
        Initialize MiniCPM-V model.

        Args:
            model_name (str): Model name, e.g., "openbmb/MiniCPM-V-2_6".
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Dynamically load the model and tokenizer based on the model name.
        """
        from transformers import AutoModel, AutoTokenizer

        if "MiniCPM-V" in self.model_name:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                attn_implementation="sdpa",  # Use SDPA attention
                torch_dtype=torch.bfloat16
            ).eval().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        else:
            raise ValueError(f"Unrecognized model name: {self.model_name}")

    def preprocess_prompt(self, instructions: list[dict]):
        """
        Preprocess the unified prompt into the model's input format.

        Args:
            instructions (list[dict]): A list of unified prompts.

        Returns:
            list[dict]: Preprocessed messages ready for inference.
        """
        messages = []
        for sample in instructions:
            user_content = sample["messages"][0]["content"]
            img_path = sample["images"][0]

            # Load and preprocess the image
            image = Image.open(img_path).convert("RGB")

            # Prepare the message
            message = {"role": "user", "content": [image, user_content]}
            messages.append(message)

        return messages

    def infer(self, inputs):
        """
        Perform inference on the preprocessed inputs.

        Args:
            inputs (list[dict]): Preprocessed messages.

        Returns:
            list[str]: Decoded output texts.
        """
        results = []
        for msgs in inputs:
            response = self.model.chat(
                image=None,  # MiniCPM-V does not require image tensors
                msgs=msgs,
                tokenizer=self.tokenizer,
            )
            results.append(response)
        return results

    def predict_with_prompt(self, instructions: list[dict]):
        """
        Perform inference using the unified prompt.

        Args:
            instructions (list[dict]): A list of unified prompts.

        Returns:
            list[str]: Decoded output texts.
        """
        # Preprocess the prompt
        inputs = self.preprocess_prompt(instructions)

        # Perform inference
        output_texts = self.infer(inputs)
        return output_texts


class MiniCPMVFactory(VLMModelFactory):
    def __init__(self, model_name: str):
        """
        Initialize MiniCPM-V factory.

        Args:
            model_name (str): Model name, e.g., "openbmb/MiniCPM-V-2_6".
        """
        self.model_name = model_name

    def create_model(self) -> VLMModelInterface:
        """
        Create a specific version of the MiniCPM-V model instance.

        Returns:
            VLMModelInterface: An instance of the MiniCPM-V model.
        """
        model = MiniCPMV(self.model_name)
        model.load_model()
        return model


if __name__ == '__main__':
    from src.aica_vlm.instructions.builder import InstructionBuilder

    # Initialize InstructionBuilder
    instruction_builder = InstructionBuilder(
        instruction_type="CES",  # Specify template type
        dataset_path="./datasets/ArtEmis",  # Dataset path
        emotion_model="CES"  # Emotion model type
    )

    # Build instructions
    instruction_builder.build()
    instructions = instruction_builder.get_instructions()

    # Specify the model name
    model_name = "openbmb/MiniCPM-V-2_6"

    # Create the model using the factory
    minicpm_factory = MiniCPMVFactory(model_name)
    minicpm_model = minicpm_factory.create_model()

    # Perform inference on the first 5 instructions
    results = minicpm_model.predict_with_prompt(instructions[:5])

    # Print the results
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}: {result}")
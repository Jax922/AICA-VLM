import torch
from PIL import Image
from src.aica_vlm.adaptation.vlm_model_interface import VLMModelInterface, VLMModelFactory

class Llava(VLMModelInterface):
    def __init__(self, model_name: str):
        """
        Initialize Llava model.

        Args:
            model_name (str): Model name, e.g., "llava-hf/llava-onevision-qwen2-7b-ov-hf" or "llava-hf/llava-v1.6-mistral-7b-hf".
        """
        self.model_name = model_name
        self.model_class = None
        self.processor_class = None

    def load_model(self):
        """
        Dynamically load the model and processor based on the model name.
        Raises:
            ValueError: If the model name is not recognized.
        """
        if "llava-onevision-qwen2-7b-ov-hf" in self.model_name:
            from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
            self.model_class = LlavaOnevisionForConditionalGeneration
            self.processor_class = AutoProcessor
        elif "llava-v1.6-mistral-7b-hf" in self.model_name:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            self.model_class = LlavaNextForConditionalGeneration
            self.processor_class = LlavaNextProcessor
        else:
            raise ValueError(f"Unrecognized model name: {self.model_name}")

        # Load the model and processor
        self.model = self.model_class.from_pretrained(
            self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to("cuda")
        self.processor = self.processor_class.from_pretrained(self.model_name)

    def preprocess_prompt(self, instructions: list[dict]):
        """
        Preprocess the unified prompt into the model's input format.

        Args:
            instructions (list[dict]): A list of unified prompts.

        Returns:
            torch.Tensor: Preprocessed inputs ready for inference.
        """
        messages = []
        for sample in instructions:
            user_content = sample["messages"][0]["content"]
            img_path = sample["images"][0]

            # Ensure the image path and content are valid
            if not isinstance(user_content, str) or not isinstance(img_path, str):
                raise ValueError("Invalid prompt format: 'messages' or 'images' is not a string.")

            # Extract the prompt text
            full_prompt = user_content.split("<image>", 1)[1].strip()
            message = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": full_prompt}
                ]
            }
            messages.append(message)

        # Generate text inputs for the processor
        texts = [
            self.processor.apply_chat_template([msg], add_generation_prompt=True)
            for msg in messages
        ]

        # Prepare inputs for the model
        images = [Image.open(sample["images"][0]).convert("RGB") for sample in instructions]
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt"
        ).to("cuda", torch.float16)

        return inputs

    def infer(self, inputs):
        """
        Perform inference on the preprocessed inputs.

        Args:
            inputs (torch.Tensor): Preprocessed inputs.

        Returns:
            list[str]: Decoded output texts.
        """
        # Generate predictions
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

        # Decode the generated outputs
        output_texts = self.processor.decode(output[0], skip_special_tokens=True)
        return [output_texts]

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


class LlavaFactory(VLMModelFactory):
    def __init__(self, model_name: str):
        """
        Initialize Llava factory.

        Args:
            model_name (str): Model name, e.g., "llava-hf/llava-onevision-qwen2-7b-ov-hf".
        """
        self.model_name = model_name

    def create_model(self) -> VLMModelInterface:
        """
        Create a specific version of the Llava model instance.

        Returns:
            VLMModelInterface: An instance of the Llava model.
        """
        model = Llava(self.model_name)
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
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

    # Create the model using the factory
    llava_factory = LlavaFactory(model_name)
    llava_model = llava_factory.create_model()

    # Perform inference on the first 5 instructions
    results = []
    for instruction in instructions[:5]:  # Example: process the first 5 instructions
        result = llava_model.predict_with_prompt([instruction])
        results.append(result)

    # Print the results
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}: {result}")
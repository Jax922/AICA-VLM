from qwen_vl_utils import process_vision_info
from src.aica_vlm.adaptation.vlm_model_interface import VLMModelInterface, VLMModelFactory

class QwenVL(VLMModelInterface):
    def __init__(self, model_name: str):
        """
        Initialize QwenVL model.

        Args:
            model_name (str): Model name, e.g., "Qwen/Qwen2-VL-7B-Instruct" or "Qwen/Qwen2.5-VL-7B-Instruct".
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
        if "Qwen2.5-VL" in self.model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            self.model_class = Qwen2_5_VLForConditionalGeneration
            self.processor_class = AutoProcessor
        elif "Qwen2-VL" in self.model_name:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.model_class = Qwen2VLForConditionalGeneration
            self.processor_class = AutoProcessor
        else:
            raise ValueError(f"Unrecognized model name: {self.model_name}")

        # Load the model and processor
        self.model = self.model_class.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
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
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": full_prompt}
                ]
            }
            messages.append(message)

        # Generate text inputs for the processor
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        # Process vision inputs (images and videos)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs.to("cuda")

    def infer(self, inputs):
        """
        Perform inference on the preprocessed inputs.

        Args:
            inputs (torch.Tensor): Preprocessed inputs.

        Returns:
            list[str]: Decoded output texts.
        """
        # Generate predictions
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        # Trim the generated IDs to remove input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the generated outputs
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts

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


class QwenVLFactory(VLMModelFactory):
    def __init__(self, model_name: str):
        """
        Initialize QwenVL factory.

        Args:
            model_name (str): Model name, e.g., "Qwen/Qwen2-VL-7B-Instruct".
        """
        self.model_name = model_name

    def create_model(self) -> VLMModelInterface:
        """
        Create a specific version of the QwenVL model instance.

        Returns:
            VLMModelInterface: An instance of the QwenVL model.
        """
        model = QwenVL(self.model_name)
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
    model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'

    # Create the model using the factory
    qwen_factory = QwenVLFactory(model_name)
    qwen_model = qwen_factory.create_model()

    # Perform inference on the first 5 instructions
    results = []
    for instruction in instructions[:5]:  # Example: process the first 5 instructions
        result = qwen_model.predict_with_prompt([instruction])
        results.append(result)

    # Print the results
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}: {result}")
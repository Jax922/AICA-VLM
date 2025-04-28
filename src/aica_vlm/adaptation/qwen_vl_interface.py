import sys

from qwen_vl_utils import process_vision_info

from aica_vlm.adaptation.vlm_model_interface import VLMModelFactory, VLMModelInterface


class QwenVL(VLMModelInterface):
    def __init__(self, model_type: str, model_path: str):
        """
        Initialize QwenVL model.

        Args:
            model_name (str): Model name, e.g., "Qwen/Qwen2-VL-7B-Instruct" or "Qwen/Qwen2.5-VL-7B-Instruct".
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model_class = None
        self.processor_class = None
        self.model = None
        self.processor = None

    def load_model(self):
        """
        Dynamically load the model and processor based on the model name.
        Raises:
            ValueError: If the model name is not recognized.
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

        # Preparation for inference
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
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
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

    def batch_inference(self, instructions: list[dict]):
        messages = []
        for instruction in instructions:
            message = self.process_instruction(instruction)
            messages.append(message)

        # Generate text inputs for the processor
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
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

        inputs = inputs.to("cuda")

        # Generate predictions
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        # Trim the generated IDs to remove input tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the generated outputs
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_texts


class QwenVLFactory(VLMModelFactory):
    def __init__(self, model_type: str, model_path: str):
        """
        Initialize QwenVL factory.

        Args:
            model_name (str): Model name, e.g., "Qwen/Qwen2-VL-7B-Instruct".
        """
        self.model_type = model_type
        self.model_path = model_path

    def create_model(self) -> VLMModelInterface:
        """
        Create a specific version of the QwenVL model instance.

        Returns:
            VLMModelInterface: An instance of the QwenVL model.
        """
        model = QwenVL(self.model_type, self.model_path)
        model.load_model()
        return model


if __name__ == "__main__":
    import json

    with open(
        "/public/home/202320163218/yxr_code/LLM_Workspace/AICA-VLM/datasets/abstract/instruction.json",
        "r",
        encoding="utf-8",
    ) as f:
        instructions = json.load(f)  # 加载 JSON 数据

    # Specify the model name
    model_name = "/public/home/202320163218/yxr_code/LLM_Workspace/AICA-VLM/models/Qwen/Qwen2.5-VL-3B-Instruct"

    # Create the model using the factory
    qwen_factory = QwenVLFactory(model_name)
    qwen_model = qwen_factory.create_model()

    for instruction in instructions:
        result = qwen_model.inference(instruction)
        print(result)

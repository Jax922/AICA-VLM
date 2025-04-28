import sys

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from aica_vlm.adaptation.vlm_model_interface import VLMModelFactory, VLMModelInterface


class Ovis(VLMModelInterface):
    def __init__(self, model_type: str, model_path: str):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None

    def load_model(self):
        if self.model_type in ["Ovis2", "Ovis1.6", "Ovis1.5"]:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=32768,
                trust_remote_code=True,
            ).cuda()
            self.text_tokenizer = model.get_text_tokenizer()
            self.visual_tokenizer = model.get_visual_tokenizer()
        else:
            raise ValueError(f"Unrecognized model name: {self.model_type}")

    def process_instruction(self, instruction: dict) -> list:
        user_content = instruction["messages"][0]["content"]
        img_path = instruction["images"][0]

        # Ensure the image path and content are valid
        if not isinstance(user_content, str) or not isinstance(img_path, str):
            raise ValueError(
                "Invalid prompt format: 'messages' or 'images' is not a string."
            )

        images = [Image.open(img_path)]
        max_partition = 9
        text = user_content.split("<image>", 1)[1].strip()
        query = f"<image>\n{text}"

        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, images, max_partition=max_partition
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to("cuda")
        attention_mask = attention_mask.unsqueeze(0).to("cuda")
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        return input_ids, pixel_values, attention_mask

    def inference(self, instruction: dict):
        input_ids, pixel_values, attention_mask = self.process_instruction(instruction)

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=512,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **gen_kwargs,
            )[0]
            output_text = self.text_tokenizer.decode(
                output_ids, skip_special_tokens=True
            )

        return output_text

    def batch_inference(self, instructions: list[dict]):
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []

        for instruction in instructions:
            input_ids, pixel_values, attention_mask = self.process_instruction(
                instruction
            )
            batch_input_ids.append(input_ids.to("cuda"))
            batch_attention_mask.append(attention_mask.to("cuda"))
            batch_pixel_values.append(
                pixel_values.to(
                    dtype=self.visual_tokenizer.dtype,
                    device=self.visual_tokenizer.device,
                )
            )

        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            [i.flip(dims=[0]) for i in batch_input_ids],
            batch_first=True,
            padding_value=0.0,
        ).flip(dims=[1])
        batch_input_ids = batch_input_ids[:, -self.model.config.multimodal_max_length :]
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [i.flip(dims=[0]) for i in batch_attention_mask],
            batch_first=True,
            padding_value=False,
        ).flip(dims=[1])
        batch_attention_mask = batch_attention_mask[
            :, -self.model.config.multimodal_max_length :
        ]

        # generate outputs
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True,
            )
            output_ids = self.model.generate(
                batch_input_ids,
                pixel_values=batch_pixel_values,
                attention_mask=batch_attention_mask,
                **gen_kwargs,
            )

        output_text = []

        for i in range(len(instructions)):
            output = self.text_tokenizer.decode(output_ids[i], skip_special_tokens=True)
            output_text.append(output)

        return output_text


class OvisFactory(VLMModelFactory):
    def __init__(self, model_type: str, model_path: str):
        self.model_type = model_type
        self.model_path = model_path

    def create_model(self) -> VLMModelInterface:
        model = Ovis(self.model_type, self.model_path)
        model.load_model()
        return model


if __name__ == "__main__":
    import json

    with open("./datasets/abstract/instruction.json", "r", encoding="utf-8") as f:
        instructions = json.load(f)

    model_name = "./models/Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_factory = OvisFactory(model_name)
    qwen_model = qwen_factory.create_model()

    for instruction in instructions:
        result = qwen_model.inference(instruction)
        print(result)

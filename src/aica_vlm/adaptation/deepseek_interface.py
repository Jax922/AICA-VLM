import sys
from typing_extensions import Self

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from aica_vlm.adaptation.vlm_model_interface import VLMModelFactory, VLMModelInterface

class DeepseekVL(VLMModelInterface):
    def __init__(self, model_type: str, model_path: str):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.processor = None
        self.visual_tokenizer = None

    def load_model(self):
        if self.model_type in ["Deepseek-VL2"]:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = self.model.to(torch.bfloat16).cuda().eval()
            self.processor = DeepseekVLV2Processor.from_pretrained(self.model_path)
            self.tokenizer = self.processor.tokenizer   
        else:
            raise ValueError(f"Unrecognized model name: {self.model_type}")

    def process_instruction(self, instruction: dict) -> list:

        user_content = instruction["messages"][0]["content"]
        img_path = instruction["images"][0]

        if not isinstance(user_content, str) or not isinstance(img_path, str):
            raise ValueError(
                "Invalid prompt format: 'messages' or 'images' is not a string."
            )

        text = user_content.split("<image>", 1)[1].strip()
        msg = f"<image>\n<|ref|>{text}<|ref|>"
        conversation = [
            {
                "role": "<|User|>",
                "content": msg,
                "images": [img_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to('cuda')

        return prepare_inputs

    def inference(self, instruction: dict):
        prepare_inputs = self.process_instruction(instruction)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=256,
            do_sample=False,
            use_cache=True
        )
        output_text = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
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


class DeepseekVLFactory(VLMModelFactory):
    def __init__(self, model_type: str, model_path: str):
        self.model_type = model_type
        self.model_path = model_path

    def create_model(self) -> VLMModelInterface:
        model = DeepseekVL(self.model_type, self.model_path)
        model.load_model()
        return model


if __name__ == "__main__":
    import json

    with open("./datasets/abstract/instruction.json", "r", encoding="utf-8") as f:
        instructions = json.load(f)

    model_name = "./models/Qwen/Qwen2.5-VL-3B-Instruct"
    qwen_factory = DeepseekVLFactory(model_name)
    qwen_model = qwen_factory.create_model()

    for instruction in instructions:
        result = qwen_model.inference(instruction)
        print(result)

# closed_model_interface.py
import base64
from io import BytesIO

from openai import OpenAI
from PIL import Image

import time
import json

class ClosedSourceAPIModel:
    def __init__(
        self, model_name: str = None, api_key: str = None, base_url: str = None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        # Initialize the client
        self.client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )

    def encode_image_to_base64(self, image_path):
        """Convert an image to a base64 string after resizing the longest side to 512 pixels."""
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            max_side = max(original_width, original_height)
            if max_side > 512:
                scale_ratio = 512 / max_side
                new_width = int(original_width * scale_ratio)
                new_height = int(original_height * scale_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_img}"

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

        image_b64_url = self.encode_image_to_base64(img_path)

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant capable of recognizing emotions in images, attributing causes to emotions, and generating emotion-based content.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": image_b64_url}},
                ],
            },
        ]

        return messages

    def inference(self, instruction: dict) -> str:
        messages = self.process_instruction(instruction)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=512, # for safe
            )
        except Exception as e:
            if 'qwen' in self.model_name:         
                # https://help.aliyun.com/zh/model-studio/error-code?spm=a2c4g.11186623.help-menu-2400256.d_2_11_0.353f6839uml0sQ
                error_code = e.code # others: message, type
                if error_code in ['data_inspection_failed', 'DataInspectionFailed']:
                    return "Error code: 400 - {'error': {'code': 'data_inspection_failed'} }" # for mark
                elif error_code in ['limit_requests', 'LimitRequests']:
                    time.sleep(2)
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=512, # for safe
                    )
                    return response.choices[0].message.content.strip()
                elif error_code in ['Arrearage']:
                    print("Error code: 400 - {'error': {'code': 'Arrearage'} }")

        return response.choices[0].message.content.strip()

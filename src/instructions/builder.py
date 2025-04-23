import template as T
from ..emotion_model import EmotionModel
import os
import random
import pandas as pd

class InstructionBuilder:
    def __init__(self, instruction_type, dataset_path, emotion_model: str):
        self.instruction_type = instruction_type
        self.instructions = []
        self.image_root_dir = os.path.join(dataset_path, "images")
        self.csv_file = os.path.join(dataset_path, "annotations.csv")
        self.emotion_model = EmotionModel(emotion_model)
        self.instruction_templates = T.instruction_templates[instruction_type]

        if self.emotion_model.model_name == "VA":
            self.instruction_tail = T.DES_tail
        else:
            self.instruction_tail = T.build_CES_tail(self.emotion_model.get_labels())

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def get_instructions(self):
        return self.instructions

    def build(self):
        df = pd.read_csv(self.csv_file)

        for idx, row in df.iterrows():
            img_name = row["img_name"]
            folder = row.get("img_folder", "")
            img_path = os.path.join(self.image_root_dir, folder, img_name)
            template = random.choice(self.instruction_templates)
            full_prompt = template.format(image_path=img_path) + " " + self.instruction_tail
            label = self._get_label_from_row(row)

            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<image>{full_prompt}"
                    },
                    {
                        "role": "assistant",
                        "content": self._format_label(label)
                    }
                ],
                "images": [img_path]
            }

            self.add_instruction(sample)

    def _get_label_from_row(self, row):
        if self.emotion_model.model_name == "VA":
            return {
                "valence": float(row["emotion_v"]),
                "arousal": float(row["emotion_a"])
            }
        else:
            return row["emotion_cat"]

    def _format_label(self, label):
        if isinstance(label, dict):
            return f"Valence: {label['valence']:.2f}, Arousal: {label['arousal']:.2f}"
        else: 
            return label

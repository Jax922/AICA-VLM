import os
import random

import pandas as pd

from aica_vlm.emotion_model import EmotionModel

from . import template as T


class InstructionBuilder:
    def __init__(self, instruction_type, dataset_path, emotion_model: str):
        self.instruction_type = instruction_type
        self.instructions = []
        self.dataset_path = dataset_path
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
        print(f"Building instructions for {self.instruction_type}...")
        df = pd.read_csv(self.csv_file)
        print(f"Loaded {len(df)} rows from {self.csv_file}")

        for idx, row in df.iterrows():
            img_name = row["img_name"]
            folder = row.get("img_folder", "")

            template = random.choice(self.instruction_templates)
            full_prompt = template + " " + self.instruction_tail
            label = self._get_label_from_row(row)

            sample = {
                "messages": [
                    {"role": "user", "content": f"<image>{full_prompt}"},
                    {"role": "assistant", "content": self._format_label(label)},
                ],
                "images": [img_name],
            }

            self.add_instruction(sample)

        output_file = os.path.join(self.dataset_path, "instruction.json")

        import json

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(self.instructions, f, indent=4)
        print(f"Instructions saved to {output_file}")

    def _get_label_from_row(self, row):
        if self.emotion_model.model_name == "VA":
            return {
                "valence": float(row["emotion_v"]),
                "arousal": float(row["emotion_a"]),
            }
        else:
            return row["emotion_cat"]

    def _format_label(self, label):
        if isinstance(label, dict):
            return f"Valence: {label['valence']:.2f}, Arousal: {label['arousal']:.2f}"
        else:
            return label

# src/config_loader.py

import os

import yaml

from .constants import SUPPORTED_MODEL_FAMILIES, SUPPORTED_TASKS


class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.model_info = {}
        self.tasks = []

    def load(self):
        """Load and validate the YAML configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Validate model info
        self.validate_model(raw_config)
        self.model_info = {
            "model_name": raw_config["model_name"],
            "model_type": raw_config["model_type"],
            "model_path": raw_config["model_path"],
        }

        # Validate tasks
        tasks = raw_config.get("tasks", [])
        if not tasks:
            raise ValueError("No tasks found in the config file.")

        for task in tasks:
            self.validate_task(task)
            self.tasks.append(task)

    def validate_model(self, config):
        """Validate model-level configuration."""
        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("Missing 'model_name' in config.")
        if not self.is_supported_model(model_name):
            raise ValueError(f"Unsupported model_name: {model_name}")

        model_type = config.get("model_type")
        if not model_type:
            raise ValueError("Missing 'model_type' in config.")

        model_path = config.get("model_path")
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

    def validate_task(self, task):
        """Validate task-level configuration."""
        task_name = task.get("task_name")
        print(task)
        if not task_name or task_name not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported or missing task_name: {task_name}")

        dataset_path = task.get("dataset_path")
        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        instruction_file = task.get("instruction_file")
        if not instruction_file or not os.path.exists(instruction_file):
            raise FileNotFoundError(
                f"Instruction file does not exist: {instruction_file}"
            )

        image_folder = task.get("image_folder")
        if not image_folder or not os.path.isdir(image_folder):
            raise FileNotFoundError(
                f"Image folder does not exist or is not a directory: {image_folder}"
            )

    def is_supported_model(self, model_name):
        """Check if model_name matches any supported model family."""
        for family in SUPPORTED_MODEL_FAMILIES:
            if model_name.startswith(family):
                return True
        return False

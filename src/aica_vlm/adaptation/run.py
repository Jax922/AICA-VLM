import json
import os

import yaml
from rich.console import Console
from rich.table import Table

from aica_vlm.adaptation.qwen_vl_interface import QwenVLFactory

# Initialize a Rich console for pretty printing
console = Console()


def run(config_path):
    """
    Main function to load the model, process tasks from the configuration, and execute inference.

    Args:
        config_path (str): Path to the YAML configuration file.
    """
    try:
        # Load the YAML configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Extract model information
        model_name = config.get("model_name")
        model_type = config.get("model_type")
        model_path = config.get("model_path")
        if not model_name or not model_type or not model_path:
            raise ValueError(
                "Model configuration is incomplete. Please check 'model_name', 'model_type', and 'model_path'."
            )

        # Print model information
        console.rule("[bold blue]Model Information")
        console.print(f"[bold green]Model Name:[/bold green] {model_name}")
        console.print(f"[bold green]Model Type:[/bold green] {model_type}")
        console.print(f"[bold green]Model Path:[/bold green] {model_path}")

        # Initialize the model factory and create the model
        console.print("[bold blue]Loading model...[/bold blue]")
        qwen_factory = QwenVLFactory(model_type, model_path)
        qwen_model = qwen_factory.create_model()
        console.print("[bold green]Model loaded successfully![/bold green]")

        # Process tasks
        tasks = config.get("tasks", [])
        if not tasks:
            raise ValueError("No tasks found in the configuration file.")

        console.rule("[bold blue]Tasks")
        for task in tasks:
            console.print(
                f"[bold yellow]Processing task:[/bold yellow] {task.get('task_name')} - {task.get('sub_task_name')}"
            )

            # Resolve paths for instruction_file and image_folder
            dataset_path = task.get("dataset_path")
            if not dataset_path or not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

            instruction_file = task.get(
                "instruction_file", os.path.join(dataset_path, "instruction.json")
            )
            if not os.path.exists(instruction_file):
                raise FileNotFoundError(
                    f"Instruction file does not exist: {instruction_file}"
                )

            image_folder = task.get(
                "image_folder", os.path.join(dataset_path, "images")
            )
            if not os.path.isdir(image_folder):
                raise FileNotFoundError(
                    f"Image folder does not exist or is not a directory: {image_folder}"
                )

            # Load instructions
            with open(instruction_file, "r", encoding="utf-8") as f:
                instructions = json.load(f)

            if not instructions:
                raise ValueError(f"Instruction file is empty: {instruction_file}")

            # Perform inference for each instruction
            results = []
            for idx, instruction in enumerate(instructions):
                console.print(
                    f"[bold cyan]Running inference for instruction {idx + 1}/{len(instructions)}...[/bold cyan]"
                )
                result = qwen_model.inference(instruction)
                results.append(result)

            # Save results to output file
            output_result_path = task.get("output_result_path", "./output/results.json")
            os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
            with open(output_result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            console.print(
                f"[bold green]Results saved to:[/bold green] {output_result_path}"
            )

        console.rule("[bold green]All tasks completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during execution:[/bold red] {e}")


if __name__ == "__main__":
    # Example usage
    config_file_path = (
        "./qwen2.5VL.yaml"  # Replace with the actual path to your YAML configuration
    )
    run(config_file_path)

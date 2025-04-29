import json
import os

import yaml
from rich.console import Console
from rich.table import Table

from aica_vlm.adaptation.instruction_load import InstructionLoader
from aica_vlm.adaptation.qwen_vl_interface import QwenVLFactory
from aica_vlm.metrics.eu_cls import EmotionClassificationMetrics

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

            # Load instructions using InstructionLoader
            instruction_loader = InstructionLoader(instruction_file, image_folder)
            instruction_loader.load()
            instructions = instruction_loader.get_instructions()
            maximum_instructions_num = task.get(
                "maximum_instructions_num", len(instructions)
            )
            instructions = instructions[:maximum_instructions_num]

            if not instructions:
                raise ValueError(f"No instructions loaded from: {instruction_file}")

            # Perform inference for each instruction
            results = []
            predictions = []
            references = []
            for idx, instruction in enumerate(instructions):
                console.print(
                    f"[bold cyan]Running inference for instruction {idx + 1}/{len(instructions)}...[/bold cyan]"
                )
                result = qwen_model.inference(instruction)
                temp_item_res = {
                    "image_path": instruction["images"][0],
                    "prompt_text": instruction["messages"][0]["content"],
                    "output_result": result,
                    "true_answer": instruction["messages"][1]["content"],
                }
                results.append(temp_item_res)

                # Collect predictions and references for metrics
                predictions.append(result)
                references.append(instruction["messages"][1]["content"])

            # Compute metrics
            # Flatten predictions to ensure they are strings
            flattened_predictions = [
                "".join(pred).strip() if isinstance(pred, list) else pred
                for pred in predictions
            ]

            # Compute metrics
            metrics_name = task.get("metrics", "EmotionClassificationMetrics")
            if metrics_name == "EmotionClassificationMetrics":
                metrics = EmotionClassificationMetrics().compute(
                    flattened_predictions, references
                )
            elif metrics_name == "EmotionRegressionMetrics":
                from aica_vlm.metrics.eu_reg import EmotionRegressionMetrics

                metrics = EmotionRegressionMetrics().compute(
                    y_pred=[float(p) for p in flattened_predictions],
                    y_true=[float(r) for r in references],
                )
            elif metrics_name == "EmotionReasoningMetrics":
                from aica_vlm.metrics.er import EmotionReasoningMetrics

                metrics = EmotionReasoningMetrics().compute(
                    flattened_predictions, references
                )
            else:
                raise ValueError(f"Unsupported metrics: {metrics_name}")

            console.print(
                f"[bold green]Metrics for task {task.get('task_name')}:[/bold green] {metrics}"
            )

            results_json_structure = {
                "task_name": task.get("task_name"),
                "sub_task_name": task.get("sub_task_name"),
                "dataset_path": task.get("dataset_path"),
                "instruction_file": task.get("instruction_file"),
                "image_folder": task.get("image_folder"),
                "results": results,
                "metrics": metrics,
            }

            # Save results to output file
            output_result_path = task.get("output_result_path", "./output/results.json")
            os.makedirs(os.path.dirname(output_result_path), exist_ok=True)
            with open(output_result_path, "w", encoding="utf-8") as f:
                json.dump(results_json_structure, f, ensure_ascii=False, indent=4)

            console.print(
                f"[bold green]Results and metrics saved to:[/bold green] {output_result_path}"
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

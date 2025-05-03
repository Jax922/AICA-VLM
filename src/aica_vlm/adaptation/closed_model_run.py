import json
import os

import yaml
from rich.console import Console
from rich.table import Table

from aica_vlm.adaptation.closed_model_interface import ClosedSourceAPIModel
from aica_vlm.adaptation.instruction_load import InstructionLoader
from aica_vlm.metrics.eu_cls import EmotionClassificationMetrics

# Initialize a Rich console for pretty printing
console = Console()


def validate_config(config):
    """
    Validate the configuration file and ensure all required fields are present.
    """
    required_fields = ["model_name", "api_key", "base_url"]
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValueError(
            f"Invalid configuration file. Missing required fields: {', '.join(missing_fields)}"
        )


def closedmodel_run(config_path):
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

        # Check if the configuration is valid: must contain model_name , api_key, baseurl, and tasks
        validate_config(config)

        # Extract model information
        model_name = config.get("model_name")

        # Print model information
        console.rule("[bold blue]Model Information")
        console.print(f"[bold green]Model Name:[/bold green] {model_name}")

        # Read key and baseurl
        api_key = config.get("api_key")
        baseurl = config.get("base_url")

        # Initialize closed model
        closedmodel = ClosedSourceAPIModel(
            model_name=model_name, api_key=api_key, base_url=baseurl
        )

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
            output_result_path = task.get("output_result_path", "./output/results.json")
            os.makedirs(os.path.dirname(output_result_path), exist_ok=True)

            # if there are results in the output file, load them and not overwrite
            if os.path.exists(output_result_path):
                with open(output_result_path, "r", encoding="utf-8") as f:
                    init_json = json.load(f)
                if "results" in init_json:
                    results = init_json["results"]
                    if len(results) > 0:
                        # print have exec number of results
                        console.print(
                            f"[bold green]Found {len(results)} results in the output file.[/bold green]"
                        )
                    else:
                        # if there are no results, create an empty list
                        init_json = {"results": []}
                        with open(output_result_path, "w", encoding="utf-8") as f:
                            json.dump(init_json, f, ensure_ascii=False, indent=4)

            # filter the instructions to only those that are not in the results
            instructions = [
                instruction
                for instruction in instructions
                if instruction["images"][0]
                not in [result["image_path"] for result in results]
            ]

            for idx, instruction in enumerate(instructions):
                console.print(
                    f"[bold cyan]Running inference for instruction {idx + 1}/{len(instructions)}...[/bold cyan]"
                )
                result = closedmodel.inference(instruction)
                temp_item_res = {
                    "image_path": instruction["images"][0],
                    "prompt_text": instruction["messages"][0]["content"],
                    "output_result": result,
                    "true_answer": instruction["messages"][1]["content"],
                }
                results.append(temp_item_res)
                # save the result to the output file
                with open(output_result_path, "w", encoding="utf-8") as f:
                    json.dump({"results": results}, f, ensure_ascii=False, indent=4)

            # Compute metrics

            # read the results from the output file
            with open(output_result_path, "r", encoding="utf-8") as f:
                all_json_result = json.load(f)

            # Collect predictions and references for metrics
            predictions = [
                result["output_result"] for result in all_json_result["results"]
            ]
            references = [
                result["true_answer"] for result in all_json_result["results"]
            ]

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

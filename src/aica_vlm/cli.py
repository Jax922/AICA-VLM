# src/aica_vlm/cli.py

import typer

# from aica_vlm.instructions import build_instruction_set
import yaml

import aica_vlm.metrics.llm_based_metrics as llm_based_metrics
from aica_vlm.adaptation.closed_model_run import closedmodel_run
from aica_vlm.adaptation.run import run
from aica_vlm.dataset import (
    build_balanced_benchmark_dataset,
    build_random_benchmark_dataset,
)
from aica_vlm.instructions import CoTInstructionBuilder, InstructionBuilder
from aica_vlm.metrics.eu_cls import compute_cls_metrics_manually

app = typer.Typer(help="AICA-VLM benchmark CLI")

dataset_app = typer.Typer(help="Dataset building CLI")
instruction_app = typer.Typer(help="Instruction generation CLI")


@app.callback()
def main():
    """AICA-VLM CLI entrypoint."""
    pass


# ========== Build Dataset ==========
@dataset_app.command("run")
def build_dataset(
    config: str = typer.Argument(..., help="YAML config path"),
    mode: str = typer.Option("random", help="Sampling mode: random or balanced"),
):
    """Build a benchmark dataset (random or balanced sampling)."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    typer.echo(f"Building dataset '{cfg['task_name']}' in mode: {mode}")
    if mode == "random":
        build_random_benchmark_dataset(
            cfg["datasets"], cfg["total_num"], cfg["output_dir"]
        )
    elif mode == "balanced":
        build_balanced_benchmark_dataset(
            cfg["datasets"], cfg["total_num"], cfg["output_dir"]
        )
    else:
        typer.echo("Unknown mode. Choose from: random, balanced")


# ========== Build Instruction ==========
@instruction_app.command("run")
def build_instruction(
    config: str = typer.Argument(..., help="YAML config path for instruction task"),
):
    """Build instructions from a benchmark dataset using templates and labels."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    instruction_type = cfg["instruction_type"]
    dataset_path = cfg["dataset_path"]
    emotion_model = cfg["emotion_model"]

    typer.echo(f"Building instructions for: {instruction_type}")
    typer.echo(f"Dataset path: {dataset_path}")
    typer.echo(f"Emotion model: {emotion_model}")

    builder = InstructionBuilder(
        instruction_type=instruction_type,
        dataset_path=dataset_path,
        emotion_model=emotion_model,
    )
    builder.build()
    typer.echo("Instruction generation completed.")


# ========== CoT Instruction ==========
@instruction_app.command("run-cot")
def build_instruction_cot(
    config: str = typer.Argument(..., help="YAML config path for CoT instruction task"),
):
    """Build CoT-style instructions from a benchmark dataset."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    instruction_type = cfg["instruction_type"]
    dataset_path = cfg["dataset_path"]
    emotion_model = cfg["emotion_model"]

    typer.echo(f"Building CoT instructions for: {instruction_type}")
    typer.echo(f"Dataset path: {dataset_path}")
    typer.echo(f"Emotion model: {emotion_model}")

    builder = CoTInstructionBuilder(
        instruction_type=instruction_type,
        dataset_path=dataset_path,
        emotion_model=emotion_model,
    )
    builder.build()
    typer.echo("CoT instruction generation completed.")


# ========== opensource models Benchmark ==========
@app.command("benchmark")
def benchmark(
    config: str = typer.Argument(..., help="Path to the YAML configuration file"),
):
    """
    Run benchmark tasks based on the provided YAML configuration file.
    """
    try:
        typer.echo(f"Loading benchmark configuration from: {config}")
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)

        # Validate configuration file
        if not cfg:
            typer.echo("Configuration file is empty or invalid.", err=True)
            raise typer.Exit(code=1)

        # Run the benchmark task using the `run` function from adaptation/run.py
        run(config)
        typer.echo("Benchmark execution completed successfully.")

    except Exception as e:
        typer.echo(f"Error during benchmark execution: {e}", err=True)
        raise typer.Exit(code=1)


# ========== closed model Benchmark ==========
#
@app.command("closedmodel-benchmark")
def closedmodel_benchmark(
    config: str = typer.Argument(..., help="Path to the YAML configuration file"),
):
    """
    Run benchmark tasks based on the provided YAML configuration file.
    """
    try:
        typer.echo(f"Loading benchmark configuration from: {config}")
        with open(config, "r") as f:
            cfg = yaml.safe_load(f)

        # Validate configuration file
        if not cfg:
            typer.echo("Configuration file is empty or invalid.", err=True)
            raise typer.Exit(code=1)

        # Run the benchmark task using the `closedmodel_run` function from adaptation/closed_model_run.py
        closedmodel_run(config)
        typer.echo("Benchmark execution completed successfully.")

    except Exception as e:
        typer.echo(f"Error during benchmark execution: {e}", err=True)
        raise typer.Exit(code=1)


# compute_cls_metrics_manually
@app.command("compute-cls-metrics-manually")
def compute_metrics(json_file: str):
    """
    Compute classification metrics manually using the provided JSON file.
    """
    try:
        typer.echo(f"Loading JSON file from: {json_file}")
        typer.echo("Computing classification metrics...")
        compute_cls_metrics_manually(json_file)

        typer.echo("Metrics computation completed successfully.")
    except FileNotFoundError:
        typer.echo(f"Error: File not found at {json_file}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        typer.echo("Error: Invalid JSON file format.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during metrics computation: {e}", err=True)
        raise typer.Exit(code=1)

    # compute the llm_based_metrics


@app.command("compute-llm-metrics")
def compute_llm_metrics(
    json_file: str = typer.Argument(..., help="Path to the JSON file"),
    task: str = typer.Option(
        "emotion-reasoning", help="Task type: reasoning or egeneration"
    ),
):
    """
    Compute LLM-based metrics using the provided JSON file.
    """
    try:
        typer.echo(f"Loading JSON file from: {json_file}")
        typer.echo(f"Task type: {task}")
        llm_based_metrics.run(json_file, task)
        typer.echo("LLM-based metrics computation completed successfully.")
    except FileNotFoundError:
        typer.echo(f"Error: File not found at {json_file}", err=True)
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        typer.echo("Error: Invalid JSON file format.", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Error during LLM-based metrics computation: {e}", err=True)
        raise typer.Exit(code=1)


app.add_typer(dataset_app, name="build-dataset")
app.add_typer(instruction_app, name="build-instruction")

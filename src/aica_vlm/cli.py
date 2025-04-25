# src/aica_vlm/cli.py

import typer

# from aica_vlm.instructions import build_instruction_set
import yaml

from aica_vlm.dataset import (
    build_balanced_benchmark_dataset,
    build_random_benchmark_dataset,
)
from aica_vlm.instructions import InstructionBuilder

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


app.add_typer(dataset_app, name="build-dataset")
app.add_typer(instruction_app, name="build-instruction")

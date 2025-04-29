import json
import os

from rich.console import Console
from rich.progress import track


class InstructionLoader:
    def __init__(self, instruction_file, image_folder):
        self.instruction_file = instruction_file
        self.image_folder = image_folder
        self.instructions = []
        self.console = Console()

    def load(self):
        """Load and validate the instruction file."""
        self.console.print("[bold blue]Starting to load instructions...[/bold blue]")

        if not os.path.exists(self.instruction_file):
            self.console.print(
                f"[bold red]Error: Instruction file not found: {self.instruction_file}[/bold red]"
            )
            raise FileNotFoundError(
                f"Instruction file not found: {self.instruction_file}"
            )

        self.console.print(
            f"[bold green]Loading instruction file:[/bold green] {self.instruction_file}"
        )
        with open(self.instruction_file, "r") as f:
            raw_instructions = json.load(f)

        if not isinstance(raw_instructions, list):
            self.console.print(
                "[bold red]Error: Instruction file must contain a list of instructions.[/bold red]"
            )
            raise ValueError("Instruction file must contain a list of instructions.")

        for instruction in track(
            raw_instructions, description="Validating instructions..."
        ):
            self.validate_instruction(instruction)
            instruction["images"][0] = os.path.join(
                self.image_folder, instruction["images"][0]
            )
            self.instructions.append(instruction)

        self.console.print(
            "[bold green]All instructions loaded successfully![/bold green]"
        )

    def get_instructions(self):
        return self.instructions

    def validate_instruction(self, instruction):
        """Validate individual instruction."""
        if not isinstance(instruction, dict):
            raise ValueError("Each instruction must be a dictionary.")

        # Required keys for the instruction
        required_keys = ["messages", "images"]
        for key in required_keys:
            if key not in instruction:
                raise ValueError(f"Missing required key in instruction: {key}")

        # Validate messages
        messages = instruction["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(
                "Messages must be a list containing at least two elements."
            )

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")
            if "role" not in message or "content" not in message:
                raise ValueError("Each message must contain 'role' and 'content' keys.")
            if message["role"] not in ["user", "assistant"]:
                raise ValueError("Message role must be either 'user' or 'assistant'.")
            if not isinstance(message["content"], str):
                raise ValueError("Message content must be a string.")

        # Validate images
        images = instruction["images"]
        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("Images must be a non-empty list.")

        for image in images:
            img_path = os.path.join(self.image_folder, image)
            if not os.path.exists(img_path):
                self.console.print(
                    f"[bold red]Error: Image file not found: {img_path}[/bold red]"
                )
                raise FileNotFoundError(f"Image file not found: {img_path}")

            # Check if the image file is 0KB
            if os.path.getsize(img_path) == 0:
                self.console.print(
                    f"[bold red]Error: Image file is empty (0KB): {img_path}[/bold red]"
                )
                raise ValueError(f"Image file is empty (0KB): {img_path}")


if __name__ == "__main__":
    instruction_file = (
        "/home/pci/yxr/AICA-VLM/datasets/benchmark/abstract/instruction.json"
    )
    image_folder = "/home/pci/yxr/AICA-VLM/datasets/benchmark/abstract/images"
    try:
        loader = InstructionLoader(instruction_file, image_folder)
        loader.load()
        instructions = loader.get_instructions()
        for instruction in instructions:
            print("Loaded instructions:", instruction)
    except Exception as e:
        Console().print(f"[bold red]Error:[/bold red] {e}")

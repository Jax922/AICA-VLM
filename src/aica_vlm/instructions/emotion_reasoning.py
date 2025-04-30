import base64
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from rich.console import Console
from rich.progress import Progress


def get_openai_client():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        print("Missing OPENAI_API_KEY or OPENAI_BASE_URL in your .env file.")
        print("Please create a .env file with the following format:\n")
        print("OPENAI_API_KEY=your-api-key")
        print("OPENAI_BASE_URL=https://your-base-url")
        sys.exit(1)

    return OpenAI(api_key=api_key, base_url=base_url)


client = get_openai_client()

console = Console()


def encode_image_to_base64(image_path):
    """base64 string"""
    with Image.open(image_path) as img:
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        base64_img = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_img}"


def generate_emotion_tasks_from_image(image_path, emotion_label, model="gpt-4o"):
    image_b64_url = encode_image_to_base64(image_path)

    system_prompt = (
        "You are building a benchmark to evaluate a vision-language model's emotional reasoning and emotional content generation abilities.\n\n"
        "You will receive an image and an emotion label.\n"
        "Your task is to produce two distinct English tasks:\n"
        "1. Emotion Reasoning: A reasoning question that asks the model to explain **why** the given emotion is being expressed in the image.\n"
        "2. Emotion-Guided Content Generation: Generate a writing instruction that asks the model to generate a short paragraph (50–100 words) using the target emotion tone, grounded in the image. Then provide an expressive answer.\n\n"
        "Format your response like this:\n"
        "Reasoning Question: ...\n"
        "Reasoning Answer: ...\n"
        "Generation Instruction: ...\n"
        "Generation Answer: ..."
    )

    user_prompt = f"The target emotion is: {emotion_label}. Use the provided image to perform the tasks."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_b64_url}},
                ],
            },
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def preprocess_csv_rows(rows, reasoning_dir):
    """
    Preprocess CSV rows to skip entries that already have corresponding JSON files in the reasoning directory.
    """
    existing_files = {
        Path(json_file).stem
        for json_file in os.listdir(reasoning_dir)
        if json_file.endswith(".json")
    }

    filtered_rows = [
        row for row in rows if Path(row["img_name"]).stem not in existing_files
    ]

    console.print(
        f"[bold green]{len(rows) - len(filtered_rows)} entries skipped as they already exist in the reasoning folder.[/bold green]"
    )
    return filtered_rows


def process_csv_row(row, dataset_path, reasoning_dir):
    """
    Process a single row from the CSV file and generate a JSON file.
    """
    img_name = row["img_name"]
    emotion = row["emotion_cat"]

    # Construct the image path
    image_path = os.path.join(dataset_path, "images", img_name)

    # Generate tasks using GPT
    result = generate_emotion_tasks_from_image(image_path, emotion)
    parsed_result = parse_generated_text(img_name, result, emotion)

    # Save the result to a JSON file
    output_file = os.path.join(reasoning_dir, f"{Path(img_name).stem}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_result, f, indent=4, ensure_ascii=False)


def parse_generated_text(img_name, raw_text, emotion):
    """
    Parse the raw generated text into a structured JSON format.
    """
    try:
        # Split the raw text into sections
        sections = raw_text.split("\n\n")
        result = {
            "messages": [
                {
                    "role": "user",
                    "content": sections[0].replace("Reasoning Question: ", "").strip(),
                },
                {
                    "role": "assistant",
                    "content": sections[1].replace("Reasoning Answer: ", "").strip(),
                },
                {
                    "role": "user",
                    "content": sections[2]
                    .replace("Generation Instruction: ", "")
                    .strip(),
                },
                {
                    "role": "assistant",
                    "content": sections[3].replace("Generation Answer: ", "").strip(),
                },
            ],
            "images": [img_name],
        }
        return result
    except IndexError:
        raise ValueError(
            "Failed to parse the generated text. Ensure the format matches the expected structure."
        )


def main(dataset_path):
    annotations_file = os.path.join(dataset_path, "annotations.csv")
    reasoning_dir = os.path.join(dataset_path, "reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)

    with open(annotations_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows = preprocess_csv_rows(rows, reasoning_dir)
    console.print(
        f"[bold green]Starting to process {len(rows)} entries from the CSV file...[/bold green]"
    )

    # Process CSV rows in parallel
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing entries...", total=len(rows))

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_csv_row, row, dataset_path, reasoning_dir)
                for row in rows
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    console.print(
                        f"[bold red]Failed to process an entry: {e}[/bold red]"
                    )
                finally:
                    progress.update(task, advance=1)

    # Merge all JSON files
    console.print("[bold green]Merging all JSON files...[/bold green]")
    all_results = []
    for json_file in Path(reasoning_dir).glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    # Save the merged JSON file
    output_file = os.path.join(dataset_path, "emotion-reasoning.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    console.print(
        f"[bold green]Processing complete! Results saved to {output_file}[/bold green]"
    )


if __name__ == "__main__":
    dataset_path = r"D:\vlm-eq\AICA-VLM\datasets\benchmark\emoset"
    main(dataset_path)

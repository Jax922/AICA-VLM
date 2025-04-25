import torch
from PIL import Image
from src.aica_vlm.adaptation.vlm_model_interface import VLMModelInterface, VLMModelFactory
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

# Constants for image normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
    Build image transformation pipeline.

    Args:
        input_size (int): Target size for resizing the image.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Dynamically preprocess an image by splitting it into blocks.

    Args:
        image (PIL.Image): Input image.
        min_num (int): Minimum number of blocks.
        max_num (int): Maximum number of blocks.
        image_size (int): Target size for each block.
        use_thumbnail (bool): Whether to add a thumbnail of the image.

    Returns:
        list[PIL.Image]: List of processed image blocks.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate target aspect ratios
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num},
        key=lambda x: x[0] * x[1]
    )

    # Find the closest aspect ratio
    target_aspect_ratio = min(
        target_ratios,
        key=lambda ratio: abs(aspect_ratio - (ratio[0] / ratio[1]))
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))

    # Split the image into blocks
    blocks = []
    for i in range(target_aspect_ratio[0] * target_aspect_ratio[1]):
        box = (
            (i % target_aspect_ratio[0]) * image_size,
            (i // target_aspect_ratio[0]) * image_size,
            ((i % target_aspect_ratio[0]) + 1) * image_size,
            ((i // target_aspect_ratio[0]) + 1) * image_size
        )
        blocks.append(resized_img.crop(box))

    # Optionally add a thumbnail
    if use_thumbnail and len(blocks) > 1:
        blocks.append(image.resize((image_size, image_size)))

    return blocks

def load_image(image_file, input_size=448, max_num=12):
    """
    Load and preprocess an image.

    Args:
        image_file (str): Path to the image file.
        input_size (int): Target size for resizing.
        max_num (int): Maximum number of blocks.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values

class InternVL(VLMModelInterface):
    def __init__(self, model_name: str):
        """
        Initialize InternVL model.

        Args:
            model_name (str): Model name, e.g., "OpenGVLab/InternVL2_5-8B" or "OpenGVLab/InternVL3-8B".
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Dynamically load the model and tokenizer based on the model name.
        """
        from transformers import AutoModel, AutoTokenizer

        if "InternVL2_5" in self.model_name or "InternVL3" in self.model_name:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True
            ).eval().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, use_fast=False)
        else:
            raise ValueError(f"Unrecognized model name: {self.model_name}")

    def preprocess_prompt(self, instructions: list[dict]):
        """
        Preprocess the unified prompt into the model's input format.

        Args:
            instructions (list[dict]): A list of unified prompts.

        Returns:
            tuple: Preprocessed questions and pixel values.
        """
        questions = []
        pixel_values = None

        for sample in instructions:
            question = sample["messages"][0]["content"]
            questions.append(question)
            img_path = sample["images"][0]
            img_pixel = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()

            if pixel_values is None:
                pixel_values = img_pixel
            else:
                pixel_values = torch.cat((pixel_values, img_pixel), dim=0)

        return questions, pixel_values

    def infer(self, inputs, batch_size=4):
        """
        Perform inference on the preprocessed inputs in batches.

        Args:
            inputs (tuple): A tuple containing questions and pixel values.
            batch_size (int): The size of each batch for inference.

        Returns:
            list[str]: Decoded output texts for all inputs.
        """
        questions, pixel_values = inputs
        num_samples = len(questions)
        results = []

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_questions = questions[start_idx:end_idx]
            batch_pixel_values = pixel_values[start_idx:end_idx]

            generation_config = dict(max_new_tokens=256, do_sample=True)
            batch_responses = self.model.batch_chat(
                self.tokenizer,
                batch_pixel_values,
                num_patches_list=[batch_pixel_values.size(0)] * len(batch_questions),
                questions=batch_questions,
                generation_config=generation_config
            )
            results.extend(batch_responses)

        return results

    def predict_with_prompt(self, instructions: list[dict]):
        """
        Perform inference using the unified prompt.

        Args:
            instructions (list[dict]): A list of unified prompts.

        Returns:
            list[str]: Decoded output texts.
        """
        inputs = self.preprocess_prompt(instructions)
        return self.infer(inputs)

class InternVLFactory(VLMModelFactory):
    def __init__(self, model_name: str):
        """
        Initialize InternVL factory.

        Args:
            model_name (str): Model name, e.g., "OpenGVLab/InternVL2_5-8B".
        """
        self.model_name = model_name

    def create_model(self) -> VLMModelInterface:
        """
        Create a specific version of the InternVL model instance.

        Returns:
            VLMModelInterface: An instance of the InternVL model.
        """
        model = InternVL(self.model_name)
        model.load_model()
        return model

# Unit Test
if __name__ == '__main__':
    from src.aica_vlm.instructions.builder import InstructionBuilder

    # Initialize InstructionBuilder
    instruction_builder = InstructionBuilder(
        instruction_type="CES",
        dataset_path="./datasets/ArtEmis",
        emotion_model="CES"
    )

    # Build instructions
    instruction_builder.build()
    instructions = instruction_builder.get_instructions()

    # Specify the model name
    model_name = "OpenGVLab/InternVL3-8B"

    # Create the model using the factory
    intern_vl_factory = InternVLFactory(model_name)
    intern_vl_model = intern_vl_factory.create_model()

    # Perform inference on the first 5 instructions
    results = intern_vl_model.predict_with_prompt(instructions[:5])

    # Print the results
    for idx, result in enumerate(results):
        print(f"Result {idx + 1}: {result}")
import base64
import io
import os
import random
import re
from enum import Enum, auto

import anthropic
import prompts.prompts as prompts
import prompts.styles as styles
import replicate
import replicate.helpers
from datasets import Dataset
from datasets import Image as HFImage
from dotenv import load_dotenv
from PIL import Image as PIL_Image

# Loads .env file into environment variable; e.g. so we can access ANTHROPIC_API_KEY
load_dotenv()

# Get an Anthropic API key on your own from the Anthropic web console.
# NOTE: I'm using Claude 3.5 Sonnet for now... but maybe we should try using an open-weights model like L3.1 8B Instruct via (eg) OpenRouter
try:
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
except KeyError:
    raise ValueError(
        "API key not found. Please set the ANTHROPIC_API_KEY environment variable."
    )

# Replicate API token isn't needed explicitly (replicate client library accesses it via environment variable)
# But if it's not present in the environment, we'd like to know about it.
try:
    os.environ["REPLICATE_API_TOKEN"]
except KeyError:
    raise ValueError(
        "REPLICATE_API_TOKEN not found. Please set the REPLICATE_API_TOKEN environment variable."
    )


class ImageCategory(Enum):
    """
    The categories of images that the language model will be prompted to generate captions for.
    The auto() function automatically assigns unique monotically-increasing integer values to each enum member.
    """

    SYNONYM_IDIOMATIC = auto()
    SYNONYM_LITERAL = auto()
    RELATED_IDIOMATIC = auto()
    RELATED_LITERAL = auto()
    DISTRACTOR = auto()


def prompt_for_image_prompts(compound: str) -> dict[ImageCategory, str]:
    """
    Prompts the language model to create image generation prompts for the given compound and usage.
    """
    prompt = prompts.PROMPT_V1.format(COMPOUND=compound)
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return extract_image_categories(response.content[0].text)


def extract_image_categories(
    response: str, enhance_with_style: bool = False
) -> dict[ImageCategory, str]:
    """
    Parses the chat response to extract the image categories and prompts.
    Args:
        response: The chat response from the API.
        enhance_with_style: Whether to enhance the prompts with the same randomly-selected style modifier.
    Returns:
        A dictionary with (image category, prompt) items.
    """

    category_prompts = {}
    category_number_name_lookup = {
        1: ImageCategory.SYNONYM_IDIOMATIC,
        2: ImageCategory.SYNONYM_LITERAL,
        3: ImageCategory.RELATED_IDIOMATIC,
        4: ImageCategory.RELATED_LITERAL,
        5: ImageCategory.DISTRACTOR,
    }

    pattern = r"<category(\d)>(.*?)</category\1>"
    matches = re.finditer(pattern, response, re.DOTALL)

    for match in matches:
        category_num = int(match.group(1))
        prompt = match.group(2).strip()
        category_prompts[category_number_name_lookup[category_num]] = prompt

    # Optionally enhance the prompts with a randomly selected style modifier.
    if enhance_with_style:
        style_modifier = f"Generate an image in the following style: {random.choice(styles.STYLE_MODIFIERS)}"
        for category in category_prompts:
            prompt = category_prompts[category]
            delimiter = " " if prompt.endswith((".", "!", "?")) else ". "
            category_prompts[category] = f"{prompt}{delimiter}{style_modifier}"

    return category_prompts


def _generate_image(prompt: str) -> bytes:
    """
    Generates an image for the given prompt.
    Returns the image as bytes instead of showing it.
    """
    model_name = "black-forest-labs/flux-schnell"
    model_input = {"prompt": prompt}

    response = replicate.run(model_name, input=model_input)

    # Unpack the single replicate.helpers.FileOutput object in the list
    file_output: replicate.helpers.FileOutput = response[0]
    return file_output.read()


def generate_images(
    category_prompts: dict[ImageCategory, str]
) -> dict[ImageCategory, bytes]:
    """
    Generates images for the given category prompts.
    """
    generated_images = {}
    for category, prompt in category_prompts.items():
        generated_images[category] = _generate_image(prompt)
    return generated_images


def create_dataset_entry(
    compound: str,
    category_prompts: dict[ImageCategory, str],
    generated_images: dict[ImageCategory, bytes],
):
    """
    Creates a dictionary entry for the dataset.
    """
    return {
        "compound": compound,
        "prompts": {cat.name: prompt for cat, prompt in category_prompts.items()},
        "images": {cat.name: img for cat, img in generated_images.items()},
    }


def create_and_push_dataset(compounds: list[str], push_to_hub: bool = True):
    """
    Creates a dataset from multiple compounds and optionally pushes it to HuggingFace Hub.
    """
    dataset_entries = []

    for compound in compounds:
        prompts = prompt_for_image_prompts(compound)
        images = generate_images(prompts)
        entry = create_dataset_entry(compound, prompts, images)
        dataset_entries.append(entry)

    # Create the dataset
    dataset = Dataset.from_list(dataset_entries)

    # Convert the image bytes to HuggingFace Image format
    dataset = dataset.cast_column("images", HFImage())

    if push_to_hub:
        dataset.push_to_hub(
            "your-username/your-dataset-name",
            token="your_huggingface_token",  # You can also use the HUGGINGFACE_TOKEN environment variable
        )

    return dataset


if __name__ == "__main__":
    # Example usage
    compounds = ["burn the midnight oil", "piece of cake", "break the ice"]
    dataset = create_and_push_dataset(compounds)

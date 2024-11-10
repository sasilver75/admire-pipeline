import base64
import io
import os
import random
import re
from datetime import datetime
from enum import Enum, auto

import anthropic
import prompts.prompts as prompts
import prompts.styles as styles
import replicate
import replicate.helpers
from datasets import Dataset, Features
from datasets import Image as HFImage
from datasets import Value
from dotenv import load_dotenv
from tqdm import tqdm

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

try:
    os.environ["HUGGINGFACE_TOKEN"]
except KeyError:
    raise ValueError(
        "HUGGINGFACE_TOKEN not found. Please set the HUGGINGFACE_TOKEN environment variable."
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


def generate_image_prompts(compound: str) -> dict[ImageCategory, str]:
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
    # TODO: What other hyperparameters can we set or fix? Surely random_seed is an important one for reproducibility?
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


def create_dataset_entries(
    compound: str,
    category_prompts: dict[ImageCategory, str],
    generated_images: dict[ImageCategory, bytes],
) -> dict:
    """
    Creates a dictionary entry for the dataset.
    Args:
        compound: The compound to use for the dataset entry.
        category_prompts: The category prompts to use for the dataset entry.
        generated_images: The generated images to use for the dataset entry (still in byte form)
    """
    entries = []

    idiomatic_ordering = [
        ImageCategory.SYNONYM_IDIOMATIC,
        ImageCategory.RELATED_IDIOMATIC,
        ImageCategory.RELATED_LITERAL,
        ImageCategory.SYNONYM_LITERAL,
        ImageCategory.DISTRACTOR,
    ]
    literal_ordering = [
        ImageCategory.SYNONYM_LITERAL,
        ImageCategory.RELATED_LITERAL,
        ImageCategory.RELATED_IDIOMATIC,
        ImageCategory.SYNONYM_IDIOMATIC,
        ImageCategory.DISTRACTOR,
    ]

    literal_sentence = "LITERAL PLACEHOLDER"  # generate_sentence(compound, idiom_type="literal")  # TODO: Implement this. Probably not here either. This should just be assembly, so pass it in as args.
    idiomatic_sentence = "IDIOMATIC PLACEHOLDER"  # generate_sentence(compound, idiom_type="idiomatic")  # TODO: Implement this. Probably not here either. This should just be assembly, so pass it in as args.

    # Create two entries for our dataset, one for each interpretation of the idiom
    for sentence, ordering_rules, sentence_type in zip(
        [literal_sentence, idiomatic_sentence],
        [literal_ordering, idiomatic_ordering],
        ["literal", "idiomatic"],
    ):
        ordered_images: list[HFImage] = [
            generated_images[category] for category in ordering_rules
        ]
        ordered_prompts: list[str] = [
            category_prompts[category] for category in ordering_rules
        ]

        # ID will be added later, once all entries have been assembled.
        # image_1 is the most relevant, based on the ordering rules
        entries.append(
            {
                "compound": compound,
                "sentence_type": sentence_type,
                "sentence": sentence,
                "image_1_prompt": ordered_prompts[0],
                "image_1": ordered_images[0],
                "image_2_prompt": ordered_prompts[1],
                "image_2": ordered_images[1],
                "image_3_prompt": ordered_prompts[2],
                "image_3": ordered_images[2],
                "image_4_prompt": ordered_prompts[3],
                "image_4": ordered_images[3],
                "image_5_prompt": ordered_prompts[4],
                "image_5": ordered_images[4],
            }
        )

    return entries


def create_and_push_dataset(compounds: list[str], push_to_hub: bool = True):
    """
    Creates a dataset from multiple compounds and optionally pushes it to HuggingFace Hub.

    Args:
        compounds: The compounds to use for the dataset.
        push_to_hub: Whether to push the dataset to HuggingFace Hub. It's likely you want to do this; the filename includes the Datetime, so we won't overwrite anything.
    """
    dataset_entries: list[dict] = []

    for compound in tqdm(
        compounds,
        desc="Generating prompts and images for compounds",
        total=len(compounds),
    ):
        # Generate the prompts for our compound, for each of the 5 image categories
        prompts = generate_image_prompts(compound)

        # Generate an image for each of the 5 categories
        images = generate_images(prompts)

        # Create two entries for our dataset, one for each interpretation of the idiom
        entries = create_dataset_entries(compound, prompts, images)

        dataset_entries.extend(entries)

    # Add an ID column to our dataset entries
    for i, entry in enumerate(dataset_entries):
        entry["id"] = i

    # Create the Dataset object from our list of dictionaries.
    print(f"Debug - type of first image: {type(dataset_entries[0]['image_1'])}")
    # Tells PyArrow what to expect for each column
    features = Features(
        {
            "id": Value("int32"),
            "compound": Value("string"),
            "sentence_type": Value("string"),
            "sentence": Value("string"),
            "image_1_prompt": Value("string"),
            "image_1": HFImage(),
            "image_2_prompt": Value("string"),
            "image_2": HFImage(),
            "image_3_prompt": Value("string"),
            "image_3": HFImage(),
            "image_4_prompt": Value("string"),
            "image_4": HFImage(),
            "image_5_prompt": Value("string"),
            "image_5": HFImage(),
        }
    )
    dataset = Dataset.from_list(dataset_entries, features=features)

    # Order the columns as we want them in the dataset on HuggingFace
    column_order = [
        "id",
        "compound",
        "sentence_type",
        "sentence",
        "image_1_prompt",
        "image_1",
        "image_2_prompt",
        "image_2",
        "image_3_prompt",
        "image_3",
        "image_4_prompt",
        "image_4",
        "image_5_prompt",
        "image_5",
    ]
    dataset = dataset.select_columns(column_order)

    # Optionally save the dataset
    if push_to_hub:
        organization_name = "UCSC-Admire"
        bonus_tag = ""  # A tag to be appended to the dataset name, if you desire
        dataset_name = f"idiom-dataset-{len(compounds)}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{f"-{bonus_tag}" if bonus_tag else ""}"

        dataset.push_to_hub(
            f"{organization_name}/{dataset_name}",
            token=os.environ["HUGGINGFACE_TOKEN"],
        )
        print(f"Dataset pushed to HuggingFace Hub: {organization_name}/{dataset_name}")

    return dataset


def get_compounds() -> list[str]:
    """
    Gets the compounds to use for the dataset.
    """
    print(f"Loading compounds...")
    compounds = ["burn the midnight oil", "piece of cake"]
    print(f"Loaded {len(compounds)} compounds.")
    return compounds


if __name__ == "__main__":
    # Get compounds
    compounds = get_compounds()

    # Create dataset using compounds
    dataset = create_and_push_dataset(compounds)

    print("Done!")

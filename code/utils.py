import os
import random
import re
from enum import Enum, auto

import anthropic
import prompts.prompts as prompts
import prompts.styles as styles
from dotenv import load_dotenv

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
    # TODO: We might want to change the values in category_prompts to be a two-tuple (prompt, style_modifier) so we can record this information later instead of needing to extract it.
    # Alternatively the prompts are structured such that we can extract teh style modifier directly pretty easily if it's present... But if style modifiers are used, they should be used in the whole dataset.
    if enhance_with_style:
        style_modifier = f"Generate an image in the following style: {random.choice(styles.STYLE_MODIFIERS)}"
        for category in category_prompts:
            prompt = category_prompts[category]
            delimiter = " " if prompt.endswith((".", "!", "?")) else ". "
            category_prompts[category] = f"{prompt}{delimiter}{style_modifier}"

    return category_prompts


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

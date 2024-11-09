from enum import Enum, auto
from dotenv import load_dotenv
import anthropic
import os
import prompts.prompts as prompts

# Loads .env file into environment variable; e.g. so we can access ANTHROPIC_API_KEY
load_dotenv()

# Get an Anthropic API key on your own from the Anthropic web console.
# NOTE: I'm using Claude 3.5 Sonnet for now... but maybe we should try using an open-weights model like L3.1 8B Instruct via (eg) OpenRouter 
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError(
        "API key not found. Please set the ANTHROPIC_API_KEY environment variable."
    )
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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

def extract_image_categories(response: str) -> dict[ImageCategory, str]:
    """
    Parses the chat response to extract the image categories and prompts.
    Args:
        response: The chat response from the API.
    Returns:
        A dictionary with (image category, prompt) items.
    """
    categories = {}
    category_number_name_lookup = {
        1: ImageCategory.SYNONYM_IDIOMATIC,
        2: ImageCategory.SYNONYM_LITERAL,
        3: ImageCategory.RELATED_IDIOMATIC,
        4: ImageCategory.RELATED_LITERAL,
        5: ImageCategory.DISTRACTOR,
    }

    for i in range(1, 6):
        category_tag = f"<category{i}>"
        end_tag = f"</category{i}>"
        
        start_idx = response.find(category_tag) + len(category_tag)
        end_idx = response.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            prompt = response[start_idx:end_idx].strip()
            categories[category_number_name_lookup[i]] = prompt
            
    return categories

def prompt_for_image_prompts(compound: str, usage: str) -> dict[ImageCategory, str]:
    """
    Prompts the language model to create image generation prompts for the given compound and usage.
    """
    prompt = prompts.PROMPT_V1.format(COMPOUND=compound, USAGE=usage)
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return extract_image_categories(response.content[0].text)
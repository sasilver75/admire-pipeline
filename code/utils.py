from enum import Enum, auto

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
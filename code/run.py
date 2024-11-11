import asyncio
import os
import random
import re
import time
from datetime import datetime
from enum import Enum, auto
from typing import NamedTuple

import prompts.prompts as prompts
import prompts.styles as styles
import replicate
import replicate.helpers
from datasets import Dataset, Features, Image, Value
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

# Load environment variables
load_dotenv()

# Verify required environment variables
for env_var in ["REPLICATE_API_TOKEN", "HUGGINGFACE_TOKEN"]:
    if env_var not in os.environ:
        raise ValueError(
            f"{env_var} not found. Please set the {env_var} environment variable."
        )

SEED = 42


class ImageCategory(Enum):
    """Categories of images for language model caption generation."""

    SYNONYM_IDIOMATIC = auto()
    SYNONYM_LITERAL = auto()
    RELATED_IDIOMATIC = auto()
    RELATED_LITERAL = auto()
    DISTRACTOR = auto()


class SentenceType(Enum):
    """Types of sentences (idiomatic vs literal)."""

    IDIOMATIC = auto()
    LITERAL = auto()


class LanguageType(Enum):
    """Types of languages, using ISO 639-1 standard language codes"""

    ENGLISH = "en"
    PORTUGUESE = "pt"
    FRENCH = "fr"
    SPANISH = "es"


class CompoundItem(NamedTuple):
    """A single compound, with its language."""

    compound: str
    langauge: LanguageType


class PromptItem(NamedTuple):
    """A single prompt for an image category, with style modifier."""

    prompt: str
    style_modifier: str


class ImageItem(NamedTuple):
    """A single image for an image category, with prompt and style modifier."""

    prompt: str
    style_modifier: str
    image: bytes


class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            # Add new tokens based on time passed since last acquire call
            now = time.time()
            new_tokens = (now - self.last_update) * self.rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_update = now

            # If we need tokens, wait for them to be added, then update token counts
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                print(f"No tokens available; waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
                # Recalculate tokens after sleep
                now = time.time()
                new_tokens = (now - self.last_update) * self.rate
                self.tokens = min(self.capacity, new_tokens)
                self.last_update = now

            # "Spend" a token
            self.tokens -= 1


# Replicate limit of 600 requests per minute = 10 requests per second
# I'm setting the capacity to 500, just so that we don't accidentally trip the rate limit somehow.
RATE_LIMITER = TokenBucket(rate=10, capacity=500)


async def generate_image_prompts_and_sentences(
    compound: CompoundItem, additional_styles: int = 0
) -> tuple[dict[ImageCategory, list[PromptItem]], dict[SentenceType, str]]:
    """
    Asynchronously generate image prompts and sentences using LLaMA.
    Args:
        compound: The compound to generate prompts and sentences for.
    Returns:
        A tuple of dictionaries:
            - The first containing (image category, promptList) pairs.
                - If extract_image_categories_and_sentences is called with enhance_with_styles=True,
                then the promptList will have multiple prompsts, each sharing the same base "content" but with different style modifiers.
            - The second containing (sentence type, sentence) pairs.
    """
    model_name = "meta/meta-llama-3-70b-instruct"
    input = {
        "prompt": prompts.USER_PROMPT.format(
            COMPOUND=compound.compound, LANGUAGE=compound.langauge.name
        ),
        "system_prompt": prompts.SYSTEM_PROMPT,
        "temperature": 0.1,
        "max_tokens": 4096,
        "seed": SEED,
    }

    # Wait for a token before making the API call
    await RATE_LIMITER.acquire()

    # Get response (Response is a list of strings)
    response = replicate.run(model_name, input=input)
    response_joined = "".join(response)

    return extract_image_categories_and_sentences(
        compound, response_joined, additional_styles=additional_styles
    )


def extract_image_categories_and_sentences(
    compound: CompoundItem, response: str, additional_styles: int = 0
) -> tuple[dict[ImageCategory, list[PromptItem]], dict[SentenceType, str]]:
    """
    Parses the chat response to extract the image categories and prompts.
    Args:
        response: The chat response from the API.
        enhance_with_style: Whether to enhance the prompts with the same randomly-selected style modifier.
    Returns:
        A dictionary with (image category, prompt) items.
    """
    sentence_prompts: dict[SentenceType, str] = {}
    category_prompts: dict[ImageCategory, list[PromptItem]] = {}
    category_number_name_lookup = {
        1: ImageCategory.SYNONYM_IDIOMATIC,
        2: ImageCategory.SYNONYM_LITERAL,
        3: ImageCategory.RELATED_IDIOMATIC,
        4: ImageCategory.RELATED_LITERAL,
        5: ImageCategory.DISTRACTOR,
    }

    # 1: Extract the example sentences for each interpretation
    sentence_pattern = r"<(idiomatic|literal)_sentence>(.*?)</\1_sentence>"
    sentence_matches = re.finditer(sentence_pattern, response, re.DOTALL)
    for match in sentence_matches:
        sentence_type = (
            SentenceType.IDIOMATIC
            if match.group(1) == "idiomatic"
            else SentenceType.LITERAL
        )
        sentence = match.group(2).strip()
        sentence_prompts[sentence_type] = sentence

    if len(sentence_prompts) != len(SentenceType):
        raise ValueError(
            f"Expected to find {len(SentenceType)} sentences, but found {len(sentence_prompts)}. Found sentences: {list(sentence_prompts.keys())}"
        )

    # 2: Extract the image prompts for each category
    category_pattern = r"<category(\d)>(.*?)</category\1>"
    category_matches = re.finditer(category_pattern, response, re.DOTALL)

    for match in category_matches:
        category_num = int(match.group(1))
        prompt = match.group(2).strip()
        prompt_item = PromptItem(prompt=prompt, style_modifier="<NONE>")
        category_prompts[category_number_name_lookup[category_num]] = [prompt_item]

    if len(category_prompts) != len(category_number_name_lookup):
        raise ValueError(
            f"Expected to find {len(category_number_name_lookup)} categories, but found {len(category_prompts)}. Found categories: {list(category_prompts.keys())}"
        )

    # 3: Optionally enhance the image prompts with the desired number of randomly-selected style modifiers.
    if additional_styles:
        # Set Random seed. By setting it here (and incorporating the compound), we ensure that each compound gets its own unique but deterministic set of styles, the same across runs.
        random_seed = hash(f"{SEED}-{compound.compound}")
        random.seed(random_seed)

        # Sample N additional styles without replacement
        if additional_styles > len(styles.STYLE_MODIFIERS):
            print(
                f"Warning: requested {additional_styles} additional styles, but only {len(styles.STYLE_MODIFIERS)} styles are available. Using all {len(styles.STYLE_MODIFIERS)} styles."
            )
        style_modifiers = random.sample(
            styles.STYLE_MODIFIERS, min(additional_styles, len(styles.STYLE_MODIFIERS))
        )

        for style_modifier in style_modifiers:
            for category in category_prompts:
                base_prompt = category_prompts[category][0].prompt
                delimiter = " " if base_prompt.endswith((".", "!", "?")) else ". "
                style_modifier_phrase = (
                    f"Generate the image in the following style: {style_modifier}."
                )
                prompt_item = PromptItem(
                    prompt=f"{base_prompt}{delimiter}{style_modifier_phrase}",
                    style_modifier=style_modifier,
                )
                category_prompts[category].append(prompt_item)

    return category_prompts, sentence_prompts


async def _generate_image(prompt: str) -> bytes:
    """Asynchronously generate an image for the given prompt."""
    model_name = "black-forest-labs/flux-schnell"
    model_input = {"prompt": prompt}

    # Wait for a token before making the API call
    await RATE_LIMITER.acquire()

    response = replicate.run(model_name, input=model_input)
    file_output: replicate.helpers.FileOutput = response[0]
    return file_output.read()


async def generate_images(
    category_prompts: dict[ImageCategory, list[PromptItem]]
) -> dict[ImageCategory, list[ImageItem]]:
    """
    Asynchronously generate images for all category prompts.
    Args:
        category_prompts: A dictionary of (image category, list of PromptItem) pairs.
    Returns:
        A dictionary of (image category, list of ImageItem) pairs.
    """
    # Create a flat list of tasks and keep track of which category/index they belong to
    tasks = []
    task_metadata = []  # Store (category, index, prompt_item) for each task

    for category, prompt_list in category_prompts.items():
        for idx, prompt_item in enumerate(prompt_list):
            tasks.append(_generate_image(prompt_item.prompt))
            task_metadata.append((category, idx, prompt_item))

    # Run all image generation tasks concurrently
    images = await asyncio.gather(*tasks)

    # Reconstruct the dictionary structure
    result: dict[ImageCategory, list[ImageItem]] = {}
    for (category, idx, prompt_item), image in zip(task_metadata, images):
        if category not in result:
            result[category] = []
        result[category].append(
            ImageItem(
                prompt=prompt_item.prompt,
                style_modifier=prompt_item.style_modifier,
                image=image,
            )
        )

    return result


async def process_compound(
    compound: CompoundItem, additional_styles: int = 0
) -> list[dict]:
    """
    Process a single compound asynchronously.
    Args:
        compound: The compound to process.
    Returns:
        A list of dataset entries for the compound.
    """
    # Generate prompts and sentences
    prompts, sentences = await generate_image_prompts_and_sentences(
        compound, additional_styles=additional_styles
    )

    # Generate images
    image_items: dict[ImageCategory, list[ImageItem]] = await generate_images(prompts)

    # Create dataset entries
    return create_dataset_entries(compound, sentences, image_items)


def create_dataset_entries(
    compound: CompoundItem,
    sentences: dict[SentenceType, str],
    image_items: dict[ImageCategory, list[ImageItem]],
) -> list[dict]:
    """
    Creates a dictionary entry for the dataset.
    Args:
        compound: The compound to use for the dataset entry.
        sentences: The sentences to use for the dataset entry.
        image_items: The image items to use for the dataset entry, which each contain the image bytes along with the prompt and style modifier.
    """
    entries = []

    # Define the two orderings of image categories, based on sentence type
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

    # Unpack the literal and idiomatic sentences
    literal_sentence = sentences[SentenceType.LITERAL]
    idiomatic_sentence = sentences[SentenceType.IDIOMATIC]

    num_styles = len(
        next(iter(image_items.values()))
    )  # Get the number of styles that were generated

    # For each style variation...
    for style_idx in range(num_styles):
        # Get the style modifier (same across all image categories at this index)
        style = next(iter(image_items.values()))[style_idx].style_modifier

        # Create two entries for our dataset, one for each interpretation of the idiom (under this style)
        for sentence, ordering_rules, sentence_type in zip(
            [literal_sentence, idiomatic_sentence],
            [literal_ordering, idiomatic_ordering],
            ["literal", "idiomatic"],
        ):
            # Get the style_idx'th item from each category in the correct order
            ordered_items = [
                image_items[category][style_idx] for category in ordering_rules
            ]

            # Create the entry: Note that image_1 is the most relevant, based on the ordering rules for the sentence type
            entries.append(
                {
                    "langauge": compound.language.value,
                    "compound": compound.compound,
                    "sentence_type": sentence_type,
                    "sentence": sentence,
                    "style": style,
                    "image_1_prompt": ordered_items[0].prompt,
                    "image_2_prompt": ordered_items[1].prompt,
                    "image_3_prompt": ordered_items[2].prompt,
                    "image_4_prompt": ordered_items[3].prompt,
                    "image_5_prompt": ordered_items[4].prompt,
                    "image_1": ordered_items[0].image,
                    "image_2": ordered_items[1].image,
                    "image_3": ordered_items[2].image,
                    "image_4": ordered_items[3].image,
                    "image_5": ordered_items[4].image,
                }
            )

    return entries


async def create_and_push_dataset(
    compounds: list[CompoundItem], additional_styles: int = 0, push_to_hub: bool = True
):
    """
    Asynchronously create dataset from multiple compounds.
    The top-level function that orchestrates the process.
    Args:
        compounds: The compounds to use for the dataset.
        push_to_hub: Whether to push the dataset to the HuggingFace Hub.
    Returns:
        The created Dataset object.
    """
    dataset_entries = []

    # Process all compounds concurrently with progress bar (Note that the process will seem to "jump", since tasks launched at the same time will likely complete at roughly the same time. It's not a smooth progress bar.)
    tasks = [
        process_compound(compound, additional_styles=additional_styles)
        for compound in compounds
    ]
    for coro in atqdm(
        asyncio.as_completed(tasks),
        total=len(compounds),
        desc=f"Processing {len(compounds)} compounds into {len(compounds) * 2 * additional_styles} records",
    ):
        entries = await coro
        dataset_entries.extend(entries)

    # Add IDs to entries
    for i, entry in enumerate(dataset_entries):
        entry["id"] = i

    # Create the Dataset object
    features = Features(
        {
            "id": Value("int32"),
            "language": Value("string"),
            "compound": Value("string"),
            "sentence_type": Value("string"),
            "sentence": Value("string"),
            "style": Value("string"),
            "image_1_prompt": Value("string"),
            "image_2_prompt": Value("string"),
            "image_3_prompt": Value("string"),
            "image_4_prompt": Value("string"),
            "image_5_prompt": Value("string"),
            "image_1": Image(),
            "image_2": Image(),
            "image_3": Image(),
            "image_4": Image(),
            "image_5": Image(),
        }
    )
    dataset = Dataset.from_list(dataset_entries, features=features)

    # Order columns
    column_order = [
        "id",
        "language",
        "compound",
        "sentence_type",
        "sentence",
        "style",
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

    # Push to hub if requested
    if push_to_hub:
        organization_name = "UCSC-Admire"
        dataset_name = f"idiom-dataset-{len(compounds)}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        dataset.push_to_hub(
            f"{organization_name}/{dataset_name}",
            token=os.environ["HUGGINGFACE_TOKEN"],
        )
        print(f"Dataset pushed to HuggingFace Hub: {organization_name}/{dataset_name}")

    return dataset


# NOTE: Commented out for testing
def get_compounds() -> list[CompoundItem]:
    """Get compounds from all language files."""
    print("Loading compounds...")
    compounds = []

    # Load compounds for each language, and accumulate them into `compounds`
    for language in LanguageType:
        filepath = f"data/{language.value}_idioms.txt"
        if os.path.exists(filepath):
            with open(filepath) as f:
                language_compounds = [
                    CompoundItem(text=line.strip(), language=language)
                    for line in f.readlines()
                    if line.strip()
                ]
                compounds.extend(language_compounds)
        else:
            # Let's skip languages that don't have an idiom file.
            print(
                f"Warning: Idiom file {filepath} not found for langauge {language}. Skipping language."
            )

    print(f"Loaded {len(compounds)} compounds across {len(LanguageType)} languages.")
    return compounds


async def main():
    """Main async function."""
    # Determines how many style variations to generate for each idiom, and how many records are generated.
    # If you have 2 compounds, and you set additional_styles=3, then you'll get 2 compounds * 2 interpretations * 3 styles = 12 records in the dataset.
    additional_styles = 3

    compounds = get_compounds()
    # TESTING DELETE THIS BELOW
    compounds = compounds[:2]
    # TESTING DELETE THIS ABOVE
    dataset = await create_and_push_dataset(compounds, additional_styles)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

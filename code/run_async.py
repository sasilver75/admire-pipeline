import asyncio
import base64
import io
import os
import random
import re
import time
from datetime import datetime
from enum import Enum, auto
from typing import Optional

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
from tqdm.asyncio import tqdm as async_tqdm

# Load environment variables
load_dotenv()

# Verify required environment variables
for env_var in ["REPLICATE_API_TOKEN", "HUGGINGFACE_TOKEN"]:
    if env_var not in os.environ:
        raise ValueError(
            f"{env_var} not found. Please set the {env_var} environment variable."
        )

SEED = 42


class TokenBucket:
    """Rate limiter using the token bucket algorithm."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens/second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.time()
            # Add new tokens based on time passed
            new_tokens = (now - self.last_update) * self.rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.last_update = time.time()
            else:
                self.tokens -= 1


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


# Create a global rate limiter for Replicate API
# Replicate limit of 600 requests per minute = 10 requests per second
# I'm setting the capacity to 500, just so that we don't accidentally trip the rate limit somehow.
RATE_LIMITER = TokenBucket(rate=10, capacity=500)


async def generate_image_prompts_and_sentences(
    compound: str,
) -> tuple[dict[ImageCategory, str], dict[SentenceType, str]]:
    """Asynchronously generate image prompts and sentences using LLaMA."""
    model_name = "meta/meta-llama-3-70b-instruct"
    input = {
        "prompt": prompts.USER_PROMPT.format(COMPOUND=compound),
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

    return extract_image_categories_and_sentences(response_joined)


def extract_image_categories_and_sentences(
    response: str, enhance_with_style: bool = False
) -> tuple[dict[ImageCategory, str], dict[SentenceType, str]]:
    """
    Parses the chat response to extract the image categories and prompts.
    Args:
        response: The chat response from the API.
        enhance_with_style: Whether to enhance the prompts with the same randomly-selected style modifier.
    Returns:
        A dictionary with (image category, prompt) items.
    """
    sentence_prompts: dict[SentenceType, str] = {}
    category_prompts: dict[ImageCategory, str] = {}
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
        category_prompts[category_number_name_lookup[category_num]] = prompt

    if len(category_prompts) != len(category_number_name_lookup):
        raise ValueError(
            f"Expected to find {len(category_number_name_lookup)} categories, but found {len(category_prompts)}. Found categories: {list(category_prompts.keys())}"
        )

    # Optionally enhance the prompts with a randomly selected style modifier.
    if enhance_with_style:
        style_modifier = f"Generate an image in the following style: {random.choice(styles.STYLE_MODIFIERS)}"
        for category in category_prompts:
            prompt = category_prompts[category]
            delimiter = " " if prompt.endswith((".", "!", "?")) else ". "
            category_prompts[category] = f"{prompt}{delimiter}{style_modifier}"

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
    category_prompts: dict[ImageCategory, str]
) -> dict[ImageCategory, bytes]:
    """Asynchronously generate images for all category prompts."""
    tasks = {
        category: _generate_image(prompt)
        for category, prompt in category_prompts.items()
    }

    # Wait for all image generation tasks to complete
    results = await asyncio.gather(*tasks.values())

    # Combine results with their categories
    return dict(zip(tasks.keys(), results))


async def process_compound(compound: str) -> list[dict]:
    """Process a single compound asynchronously."""
    # Generate prompts and sentences
    prompts, sentences = await generate_image_prompts_and_sentences(compound)

    # Generate images
    images = await generate_images(prompts)

    # Create dataset entries
    return create_dataset_entries(compound, sentences, prompts, images)


def create_dataset_entries(
    compound: str,
    sentences: dict[SentenceType, str],
    category_prompts: dict[ImageCategory, str],
    generated_images: dict[ImageCategory, bytes],
) -> list[dict]:
    """
    Creates a dictionary entry for the dataset.
    Args:
        compound: The compound to use for the dataset entry.
        sentences: The sentences to use for the dataset entry.
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

    literal_sentence = sentences[SentenceType.LITERAL]
    idiomatic_sentence = sentences[SentenceType.IDIOMATIC]

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
        # NOTE: image_1 is the most relevant, based on the ordering rules
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


async def create_and_push_dataset(compounds: list[str], push_to_hub: bool = True):
    """Asynchronously create dataset from multiple compounds."""
    dataset_entries = []

    # Process all compounds concurrently with progress bar (Note that the process will seem to "jump", since tasks launched at the same time will likely complete at roughly the same time. It's not a smooth progress bar.)
    tasks = [process_compound(compound) for compound in compounds]
    for coro in tqdm(
        asyncio.as_completed(tasks),
        total=len(compounds),
        desc=f"Processing {len(compounds)} compounds",
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

    # Order columns
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


def get_compounds() -> list[str]:
    """Get the compounds to use for the dataset."""
    print("Loading compounds...")
    compounds = [
        "burn the midnight oil",
        "piece of cake",
        "bite off more than you can chew",
        "raining cats and dogs",
        "kill two birds with one stone",
        "spill the beans",
        "under the weather",
    ]
    print(f"Loaded {len(compounds)} compounds.")
    return compounds


# TODO: CONFIRM THA THE TOKEN COUNT IS BEING DECREASED WHENEVER A REQUEST IS LAUNCHED! NOT WHEN WE START PROCESSING AN IDIOM! BECAUSE WE WANT RETRIES TO TAKE TOKENS OUT OF THE BUCKET TOO!
async def main():
    """Main async function."""
    compounds = get_compounds()
    dataset = await create_and_push_dataset(compounds)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

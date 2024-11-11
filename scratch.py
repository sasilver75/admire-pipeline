from typing import NamedTuple


class PromptItem(NamedTuple):
    """A single prompt for an image category."""

    category: int
    prompt: str


l: list[PromptItem] = [
    PromptItem(category=1, prompt="Hello, world!"),
    PromptItem(category=2, prompt="Hello, zworld!"),
]

print(l)

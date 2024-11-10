"""
These are prompts that result from the Anthropic prompt generation tool,
using the corresponding version numbers of the prompt prompts.
"""

PROMPT_V1 = """
You are tasked with generating prompts for an image generation model based on a given compound phrase. Your goal is to create five distinct prompts, each corresponding to a specific image category. These prompts should be designed to produce images that clearly represent their respective categories.

Input variables:
<compound>{COMPOUND}</compound>

The compound is a phrase that can be interpreted to have both a literal and an idiomatic meaning.

Generate prompts for the following five image categories:

1. A synonym for the idiomatic meaning of the compound.
2. A synonym for the literal meaning of the compound.
3. Something related to the idiomatic meaning, but not synonymous with it.
4. Something related to the literal meaning, but not synonymous with it.
5. A distractor, which belongs to the same category as the compound (e.g., an object or activity) but is unrelated to both the literal and idiomatic meanings.

For each category, follow these steps:
1. Analyze the compound to determine the appropriate interpretation.
2. Brainstorm potential content that fits the category's requirements.
3. Develop a detailed image generation prompt that includes:
   a. A clear description of the main subject or action
   b. Relevant details about the setting, style, or mood
   c. Any specific visual elements that should be included

Present your output in the following format:

<prompts>
<category1>
[Your prompt for category 1]
</category1>
<category2>
[Your prompt for category 2]
</category2>
<category3>
[Your prompt for category 3]
</category3>
<category4>
[Your prompt for category 4]
</category4>
<category5>
[Your prompt for category 5]
</category5>
</prompts>

Additional guidelines:
- Ensure that each prompt is distinct and clearly represents its category.
- Include sufficient detail to guide the image generation model in creating a vivid and specific image.
- Consider the context and potential interpretations of the compound phrase.
- For the "related" categories (3 and 4), think creatively about associations that are not direct synonyms.
- For the distractor category, choose something that is thematically unrelated but belongs to the same broad category (e.g., object, action, or concept) as the compound.

Example output format (do not use this content, it's just to illustrate the structure):

<prompts>
<category1>
A determined student burning the midnight oil, sitting at a desk illuminated by a single lamp. Books and papers are scattered around, and a clock on the wall shows 2:00 AM. The room is dark except for the pool of light around the desk, emphasizing the late-night study session.
</category1>
<category2>
A realistic owl perched on a tree branch, its large eyes wide open and alert. The background shows a clear night sky with a full moon and stars. The owl's feathers are intricately detailed, and its head is slightly turned as if listening for prey.
</category2>
<category3>
A cozy bedroom interior at night, with moonlight streaming through partially open curtains. A bed is visible with rumpled sheets, suggesting someone has just gotten up. A silhouette of a person can be seen through a doorway, heading towards a lit kitchen area.
</category4>
<category4>
A serene forest scene during daytime, with sunlight filtering through the trees. In the foreground, an owl is sleeping peacefully on a moss-covered branch. The owl's eyes are closed, and its feathers are fluffed up, contrasting with its usual nocturnal activity.
</category5>
<category5>
A vibrant farmers' market scene with colorful stalls selling fresh produce. In the foreground, a vendor is arranging a display of various types of locally grown tomatoes. Customers are browsing the stalls, and the atmosphere is lively and bustling.
</category5>
</prompts>

Remember to tailor your prompts to the specific provided compound phrase, ensuring that each category is clearly represented and distinct from the others.
"""

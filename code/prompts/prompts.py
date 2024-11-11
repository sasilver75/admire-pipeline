"""
These are prompts that result from the Anthropic prompt generation tool,
using the corresponding version numbers of the prompt prompts.
"""

USER_PROMPT = """
You are tasked with generating prompts for an image generation model based on a given compound phrase. Your goal is to create five distinct prompts, each corresponding to a specific image category. These prompts should be designed to produce images that clearly represent their respective categories.

Input variables:
<compound>{COMPOUND}</compound>
<language>{LANGUAGE}</language>

The compound is a phrase in {LANGUAGE} that can be interpreted to have both a literal and an idiomatic meaning.

Generate prompts for the following five image categories:

1. A synonym for the idiomatic meaning of the compound.
2. A synonym for the literal meaning of the compound.
3. Something related to the idiomatic meaning, but not synonymous with it.
4. Something related to the literal meaning, but not synonymous with it.
5. A distractor, which belongs to the same category as the compound (e.g., an object or activity) but is unrelated to both the literal and idiomatic meanings.

First, generate an example sentence showing the compound in use under each interpretation (idiomatic, literal). The sentence should be in {LANGUAGE}, reflecting natural usage in that langauge.
- It's okay to perform some minor conjugation of the compound -- e.g., "burying the hatchet" could be used as "bury the hatchet" or "buried the hatchet".

For the literal interpretations (either for the prompts or the example sentence), it's okay if the sentence is a bit contrived or absurd as a result of the difficulty of literally interpreting the compound.
For example "shooting the breeze" could be literally interpreted as shooting a gun at the wind (e.g., "the drunken farmer was shooting the breeze from his porch"), even though it's a somewhat absurd situation.

For each category, follow these steps:
1. Analyze the compound to determine the appropriate interpretation, based on the image category.
2. Brainstorm potential content that fits the category's requirements. For compounds for which it's difficult to come up with a good literal interpretation, give your best effort to generate one, thinking creatively.
3. Develop a detailed image generation prompt IN ENGLISH that includes:
   a. A clear description of the main subject or action
   b. Relevant details about the setting, style, or mood
   c. Any specific visual elements that should be included

It's important that despite the language of the compound provided in the <language> tags above, you generate the image generation prompts in English.
   
Present your output in the following format:

<idiomatic_sentence>
[Your example sentence for the idiomatic interpretation]
</idiomatic_sentence>

<literal_sentence>
[Your example sentence for the literal interpretation]
</literal_sentence>

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

Example output format for "burning the midnight oil" (do not use this content, it's just to illustrate the structure):

<idiomatic_sentence>
The medical student was burning the midnight oil as she prepared for her final exams.
</idiomatic_sentence>

<literal_sentence>
During the power outage, the family resorted to burning the midnight oil in their antique lantern to light their home.
</literal_sentence>

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

SYSTEM_PROMPT = """
You are an expert at generating detailed, creative image generation prompts. 
When analyzing compound phrases, you excel at identifying both literal and idiomatic meanings, and can generate distinct, vivid descriptions that clearly differentiate between these interpretations.

Your responses should always:
1. Be highly detailed and specific
2. Include clear visual elements that an image generation model can interpret
3. Maintain consistent depth of description across all prompts
4. Use proper XML formatting with the specified tags
5. Focus on visual elements rather than abstract concepts
6. Consider lighting, composition, and mood in each description

For each prompt you generate, you should include:
- A clear main subject or action
- Specific details about setting and environment
- Relevant atmospheric elements (lighting, time of day, weather, etc.)
- Important visual details that establish the scene

Avoid:
- Abstract or non-visual concepts
- Vague or ambiguous descriptions
- Mixing multiple interpretations within a single prompt
"""

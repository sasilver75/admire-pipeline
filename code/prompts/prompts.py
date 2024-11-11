"""
These are prompts that result from the Anthropic prompt generation tool,
using the corresponding version numbers of the prompt prompts.
"""

# Examples of good/bad image prompts taken from ideogram.ai's "Magic Prompt" page in their documentation.
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

For each category, develop a detailed image generation prompt IN ENGLISH. Each prompt should paint a vivid picture that clearly represents the compound in the context of the image category. For compounds where it's difficult to come up with a good literal interpretation, think creatively about how to visualize it.

Instead of generating too-simple prompts like "Photo of cupcake and strawberries on a table", make your prompts as detailed as possible, incorporating elements like:
- Subject details and positioning
- Lighting and atmosphere
- Composition and framing
- Color palette and mood
- Environmental context and background elements
- Textures and materials
- Character expressions and poses (where applicable)
- Time of day and weather conditions (where relevant) 

For example, instead of "A cat on the right side of a beach ball on a sunny beach," produce something more like:
"A playful scene featuring a graceful Siamese cat positioned on the right third of the composition, sitting next to a large beach ball with rainbow-colored stripes. The pristine white sand beach stretches into the distance, where azure waves gently roll toward the shore. Golden late afternoon sunlight bathes the scene in warm tones, creating long shadows that stretch across the sand. The cat's blue eyes catch the light while its cream and chocolate-point fur ruffles slightly in the ocean breeze. Wispy cirrus clouds streak across the vibrant sky, while distant seabirds soar over the horizon. Palm fronds sway gently at the edges of the scene, framing the composition."
Another example, instead of "Photo of cupcake and strawberries", write:
"A decadent display featuring a perfectly crafted cupcake crowned with swirls of ivory buttercream frosting and garnished with fresh strawberries. The cupcake sits centered on an elegant marble surface, with the rich chocolate base visible beneath the cloud-like frosting. A fan of precisely sliced strawberry adorns the top, their vibrant red color contrasting beautifully with the creamy frosting. Scattered around the base are additional ripe strawberries and delicate sprinklings of edible gold leaf, creating visual interest and balance. Soft, directional lighting from the left highlights the texture of the frosting while casting subtle shadows that emphasize the depth and dimension of the scene. The background fades into a pleasing blur of soft greys and whites, allowing the centerpiece to command attention."

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
- For the "related" categories (3 and 4), think creatively about associations that are not direct synonyms.
- For the distractor category, choose something that is thematically unrelated but belongs to the same broad category (e.g., object, action, or concept) as the compound.

Example output format for "burning the midnight oil" (do not use this content, it's just to illustrate the structure. The prompts you generate should be more high-quality and descriptive):

<idiomatic_sentence>
The medical student was burning the midnight oil as she prepared for her final exams.
</idiomatic_sentence>

<literal_sentence>
During the power outage, the family resorted to burning the midnight oil in their antique lantern to light their home.
</literal_sentence>

<prompts>
<category1>
A dedicated student works late into the night at a wooden desk crowded with textbooks, notebooks, and scattered papers. Heavy dark circles under their eyes betray exhaustion, yet they remain focused on their studies. A single desk lamp creates a warm pool of light that illuminates their workspace, while the rest of the room fades into shadow. Through a nearby window, a crescent moon hangs in a star-filled sky, and a digital clock on the desk prominently displays 3:27 AM. A half-empty coffee mug and crumpled energy drink cans suggest many hours of continuous work.
</category1>
<category2>
An atmospheric scene of a Victorian-era family gathered around an ornate brass oil lamp casting flickering light in their dimly lit parlor. The lamp's flame dances inside its glass chimney, creating dramatic shadows on the antiqued wallpaper. A ceramic vessel labeled "Midnight Oil" sits nearby, its contents being carefully poured into the lamp by the father figure. The warm, golden light illuminates the concerned faces of family members dressed in period-appropriate attire, while beyond the lamp's glow, the room lies in complete darkness, suggesting a power outage or pre-electricity era.
</category2>
<category3>
A cozy 24-hour diner scene at 2 AM, with several patrons visible through large windows that reflect the neon "OPEN" sign's glow. Inside, a student surrounded by textbooks sips coffee at a worn formica counter, while a waitress refills another customer's cup. The fluorescent lighting creates a harsh yet familiar ambiance, and a wall clock clearly shows the late hour. Steam rises from the coffee cups, and plates of half-eaten pie suggest the long night ahead.
</category3>
<category4>
A richly detailed scene of an old-fashioned whale oil processing facility on a moonlit harbor. Wooden barrels marked "Whale Oil" line the weathered dock, while workers in period clothing transfer the precious fuel into storage containers. The full moon reflects off the dark water, and in the distance, tall ships' masts cut silhouettes against the night sky. Lanterns hanging from posts cast an eerie glow over the nocturnal operation.
</category4>
<category5>
A vibrant bakery kitchen scene where a pastry chef expertly decorates an elaborate wedding cake. The multi-tiered cake stands on a rotating platform, while the chef applies delicate fondant flowers with precision tools. Surrounding the workspace are bowls of colorful frostings, various piping bags, and scattered sugar flowers. Natural light streams through large windows, highlighting the cake's pearlescent finish and the concentrated expression on the chef's face.
</category5>
</prompts>

Remember to tailor your prompts to the specific provided compound phrase, ensuring that each category is clearly represented and distinct from the others.
"""

SYSTEM_PROMPT = """
You are a specialized prompt engineer for image generation models. Your role is to:
1. Analyze compound phrases and their literal/idiomatic meanings
2. Generate natural example sentences showing different interpretations
3. Create detailed, vivid image generation prompts that work across different artistic styles
4. Ensure each prompt is distinct and clearly represents its assigned image category

Your outputs should be detailed enough to guide image generation while remaining style-agnostic (avoiding technical photography terms or specific medium references). 
Always maintain the exact structure of the compound phrase provided, making only minimal grammatical adjustments when necessary for sentence construction.

Avoid mixing multiple interpretations within a single prompt.
"""

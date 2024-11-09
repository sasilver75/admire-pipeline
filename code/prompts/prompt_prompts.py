
"""
These are prompts that are used with Anthropic's prompt generation tool to create the prompts that we _actually_ want to use.
We're keeping them versioned so that we have a record of the progression of the prompt prompts.
"""

# TODO: Maybe this would be better if for the examples I restated somehow (eg on a different line) what the category was, instead of relying on the LM's ability to match up each image description to the image category.
PROMPT_PROMPT_V1 = """
I want to generate a prompt that will be used to generate prompts to a diffusion model like Stable Diffusion. 

The context is that I will want to inject into this prompt a "compound" and a "usage", where a compound can be a phrase like "heart of gold," and usage has values of either "literal" or "idiomatic". 

I want this prompt to generate five outputs that can be used as prompts that can be used with my downstream image generation model. Each of these five outputs is related to one of 5 specific "image categories." I want to generate an image generation prompt for each of these five categories.

The categories are:
1. A synonym for the idiomatic meaning of the compound.
2. A synonym for the literal meaning of the compound.
3. Something related to the idiomatic meaning, but not
synonymous with it.
4. Something related to the literal meaning, but not synonymous with it.
5. A distractor, which belongs to the same category as the compound (e.g. an object or activity) but is unrelated to both the literal and idiomatic meanings.

I want the prompt to make it obvious to the language model to generate outputs in a way there it's obvious to me which image category each prompt belongs to.

I can't give any good examples of what I want the generated image-generation prompts to look like, but I can give some descriptions of some sets of images that have resulted from this process (which I'm trying to replicate):

For compound "elbow grease" and the usage of "idiomatic" (with the idiomatic interpretation of "elbow grease" usually referring to "hard work"), it seems like the following images have been generated (these are own descriptions, but the prompts I expect you to generate should likely be more detailed), for each respective image category:
1. An dirty electric stovetop has a person's arm rubbing a sponge/brillo pad over it, as if to clean it.
2. Someone's upper body is wearing a tee shirt, with one hand holding a bowl of orange grease, and the other hand applying it to their opposing elbow.
3. A roof gutter is being cleaned by the arm of a person that is equipped with a bristle-brush cleaning tool.
4. A gloved work glove is holding a can opener opening a jar of yellow grease.
5. The lower half of a person's body. They're wearing shorts and work boots, with knee pads over their knees.

Here's another example.
For the compound "night owl" and the usage of "idiomatic" (with the idiomatic interpretation of "night owl" usually referring to someone who works late into the night):
1. A boy sitting a desk, writing an essay. Above him, a pendant lamp illumates his workstation. We can see that it's nighttime outside.
2. An owl sitting a branch at night, with a full moon in the background.
3. A boy sleping on a couch, assumedly at night time.
4. An owl sitting on a branch by a lake surrounded by trees. It is day time.
5. A weightlifting dumbbell with a stainless steel handle and black weight bumpers.

Here's another example.
For the compound "heart of gold" and the usage of "idiomatic", with the idiomatic interpretation of "heart of gold" usually referring to the quality of having a kind and good nature.
1. A boy is actively laying out bowls of dog food on the ground, which are being eaten from by 4 hungry puppies. It seems that they might be in a rough area of town.
2. An anatomic human heart that appears to be made out of gold.
3. A boy and his dog happily rejoice over a christmas present that has just been opened.
4. A large gold bar that's visible inside a bank vault that has a circular door which is slightly ajar.
5. A futuristic spaceship that has a somewhat fish-like design.

Here's another example.
For the compound "piece of cake" and the usage of "literal", with the idiomatic interpretation of "piece of cake" usually referring to something that is easy to do.
1. A boy playing golf, standing with his golf club next to his ball on the green, which is inches from the hole, representing an easy shot.
2. A triangular-shaped slice of vanilla-frosted cake.
3. A man with climbing shoes and harness climbing a rock wall at a climbing gym.
4. An entire vanilla-frosted cake, with no slices taken out of it. Chocolate frosting drips over the edges of the top of the cake.
5. A bowl of Ramen soup, with noodles, pork, egg, mushrooms, scallions, and corn visible in the broth.

See that each of the image categories is represented in the generated images.
Observe how the content for the "something related to" image categories relates to their respective "A synonym for" image categories.

I do not expect for the above to be exemplars that make it into the final prompt that you generate (unless you see a reason to incorporate them), I share them moreso to explain the content of the images for each category, which might then inform the prompt you generate
In fact, I don't think that these images given as examples necessarily constitute _good_ images for the respective image categories.
I think that the prompt we generate together should likely provoke the langauge model to generate some reasoning about what the content of the image should be, based on the compound, its usage, and the image category.
Then, the model should generate an image generation prompt centered around that content, with additional details to help specify the style/specific details of the image.
"""
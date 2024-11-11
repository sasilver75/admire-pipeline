
[] Add retries, etc.
[] Get some portuguese idioms
[] Merge in the english idioms from the dataset
[] Prompt needs improvement re: materialization of reasoning of what the prompt should look like. We had some bad reuslts for (eg) a bird in the hand is worth two in the bush.
[] Question: Should we determine whether a compound OUGHT to have both interpretations?
[] Question: Should we try injecting the style differently?
[] TODO: Look for what some good Flux prompts look like. Note that "good" doesn't mean artistic. We just want control and minimal diffusion artifacts.


Notes
- It doesn't seem that HF has a limit on the number of datasets that can be created, and it seems like a dataset can be 300GB without needing permission (free).
- I have $500 in replicate credits, lol.


Desired columns
- id
- compound
- sentence_type
- sentence
- image_1
- image_2
- image_3
- image_4
- image_5

The question is whether we want to have two rows, one for for each interpretation of each idiom?
Seems like a bit of duplicated data, but HuggingFace allows for us to store a shit ton of data, so it's no worry.


Rate limit notes
- Replicate: You can create predictions at 600 requests per minute. All other endpoints you can call at 3000 requests per minute.
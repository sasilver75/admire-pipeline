[] Why does our prompt even ask for usage? We're going to be using the same prompt for both literal and idiomatic images, aren't we? For the finetuning, the thing that will differ is the sentence that is given as example (from which the VLM has to determine the sentence type, to then determine the ranking of images).
[] Should we ask the LM in the prompt to first generate an idiomatic and a literal usage? We can multi-shot prompt this using the examples from the training set.

[] I have basically no error handling in my utils file, and it's also all synchronous. If we want to scale this up, we'll need to make this async and add error handling.

[] We need an example of the idiom being used in a sentence too... both literally and figuratively. Maybe update the prompt?


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
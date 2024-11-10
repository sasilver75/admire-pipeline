

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
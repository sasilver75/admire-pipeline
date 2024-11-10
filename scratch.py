from datasets import load_dataset

dataset = load_dataset("UCSC-Admire/idiom-dataset-2-2024-11-09_22-04-33")
# To view an image
print(dataset["train"][0])  # This will show you the image data

import os
import pandas as pd


"""
The point of this is basically to assemble the Training Dataset and save it to HuggingFace.
"""

def main():
    df = pd.read_csv("data/subtask_a_train.tsv", sep="\t")
    print(f"df.shape: {df.shape}")

if __name__ == "__main__":
    main()
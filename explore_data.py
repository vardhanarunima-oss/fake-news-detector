import pandas as pd

# LIAR has no header row, so we name the columns ourselves
# There are 14 columns total — we only care about col 1 (label) and col 2 (statement)
columns = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party", "barely_true_count", "false_count",
    "pants_fire_count", "half_true_count", "mostly_true_count", "context"
]

train_df = pd.read_csv("data/train.tsv", sep="\t", header=None, names=columns)

print("Shape:", train_df.shape)
print("\nLabel distribution:")
print(train_df["label"].value_counts())
print("\nSample statements:")
print(train_df[["label", "statement"]].head(10))
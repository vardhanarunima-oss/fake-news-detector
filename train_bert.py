import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import os

# ── 1. Check if GPU is available ──────────────────────────────────────────────
# PyTorch can run on CPU or GPU. GPU is 10-20x faster for BERT.
# Most Windows laptops don't have a compatible GPU so this will likely say CPU.
# That's fine — training will take longer but will work.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 2. Load data ──────────────────────────────────────────────────────────────
columns = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party", "barely_true_count", "false_count",
    "pants_fire_count", "half_true_count", "mostly_true_count", "context"
]

train_df = pd.read_csv("data/train.tsv", sep="\t", header=None, names=columns)
valid_df = pd.read_csv("data/valid.tsv", sep="\t", header=None, names=columns)
test_df  = pd.read_csv("data/test.tsv",  sep="\t", header=None, names=columns)

# ── 3. Map labels to 0 and 1 ──────────────────────────────────────────────────
# BERT expects numbers, not strings. 0 = FAKE, 1 = REAL.

label_map = {
    "pants-fire":  0,
    "false":       0,
    "barely-true": 0,
    "half-true":   1,
    "mostly-true": 1,
    "true":        1
}

for df in [train_df, valid_df, test_df]:
    df["binary_label"] = df["label"].map(label_map)

train_df = train_df.dropna(subset=["binary_label"])
valid_df  = valid_df.dropna(subset=["binary_label"])
test_df   = test_df.dropna(subset=["binary_label"])

train_df["binary_label"] = train_df["binary_label"].astype(int)
valid_df["binary_label"]  = valid_df["binary_label"].astype(int)
test_df["binary_label"]   = test_df["binary_label"].astype(int)

print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")

# ── 4. Tokenizer ──────────────────────────────────────────────────────────────
# The tokenizer converts raw text into numbers that BERT understands.
# It breaks text into "tokens" (words or word-pieces) and maps each to an ID.
# max_length=128 means we cap at 128 tokens — enough for short statements.
# padding=True adds zeros to shorter sequences so all inputs are same length.
# truncation=True cuts longer sequences down to 128.

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# ── 5. PyTorch Dataset ────────────────────────────────────────────────────────
# PyTorch needs data in a specific format — a Dataset class that returns
# one sample at a time. DataLoader then batches these for training.

class LiarDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"  # return PyTorch tensors, not numpy
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

print("Tokenizing data... (this takes a minute)")
train_dataset = LiarDataset(train_df["statement"], train_df["binary_label"], tokenizer)
valid_dataset = LiarDataset(valid_df["statement"], valid_df["binary_label"], tokenizer)
test_dataset  = LiarDataset(test_df["statement"],  test_df["binary_label"],  tokenizer)

# batch_size=16 means we process 16 statements at once during training
# smaller batch = less RAM needed, fine for CPU training
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)
test_loader  = DataLoader(test_dataset,  batch_size=16)

# ── 6. Load DistilBERT model ──────────────────────────────────────────────────
# num_labels=2 tells it we have 2 classes (FAKE=0, REAL=1)
# This downloads ~250MB on first run, cached after that

print("Loading DistilBERT... (downloads ~250MB on first run)")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model.to(device)

# ── 7. Optimizer ──────────────────────────────────────────────────────────────
# AdamW is the standard optimizer for BERT fine-tuning.
# lr=2e-5 (0.00002) is the standard learning rate for BERT — don't change this.
# Too high and you'll destroy the pre-trained weights. Too low and it won't learn.

optimizer = AdamW(model.parameters(), lr=2e-5)

# ── 8. Training loop ──────────────────────────────────────────────────────────
# We train for 3 epochs. Each epoch = one full pass over all training data.
# On CPU this will take 2-4 hours total. Let it run — don't close the terminal.

EPOCHS = 3
best_val_accuracy = 0

print("\nStarting training...")
print("On CPU this takes 2-4 hours. Let it run.\n")

for epoch in range(EPOCHS):
    # ── Training phase ──
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()                          # clear previous gradients
        outputs = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs.loss
        loss.backward()                                # compute gradients
        optimizer.step()                               # update weights

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Print progress every 100 batches so you know it's running
        if (batch_idx + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    train_acc = accuracy_score(all_labels, all_preds)
    avg_loss  = total_loss / len(train_loader)

    # ── Validation phase ──
    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():  # no gradient computation needed for validation
        for batch in valid_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model based on validation accuracy
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        model.save_pretrained("models/bert_model")
        tokenizer.save_pretrained("models/bert_model")
        print(f"  ✓ Best model saved (val acc: {val_acc:.4f})")

# ── 9. Final test evaluation ──────────────────────────────────────────────────
print("\nLoading best model for final test evaluation...")
best_model = DistilBertForSequenceClassification.from_pretrained("models/bert_model")
best_model.to(device)
best_model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        outputs        = best_model(input_ids=input_ids, attention_mask=attention_mask)
        preds          = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(test_labels, test_preds)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print("Model saved to models/bert_model/")
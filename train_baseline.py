import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load data ──────────────────────────────────────────────────────────────
columns = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party", "barely_true_count", "false_count",
    "pants_fire_count", "half_true_count", "mostly_true_count", "context"
]

train_df = pd.read_csv("data/train.tsv", sep="\t", header=None, names=columns)
valid_df = pd.read_csv("data/valid.tsv", sep="\t", header=None, names=columns)
test_df  = pd.read_csv("data/test.tsv",  sep="\t", header=None, names=columns)

# ── 2. Map 6 labels → 2 labels ────────────────────────────────────────────────
# Fake = pants-fire, false, barely-true  |  Real = half-true, mostly-true, true
label_map = {
    "pants-fire":   "FAKE",
    "false":        "FAKE",
    "barely-true":  "FAKE",
    "half-true":    "REAL",
    "mostly-true":  "REAL",
    "true":         "REAL"
}

for df in [train_df, valid_df, test_df]:
    df["binary_label"] = df["label"].map(label_map)

# Drop any rows where label wasn't in our map (shouldn't happen, but safety)
train_df = train_df.dropna(subset=["binary_label"])
valid_df  = valid_df.dropna(subset=["binary_label"])
test_df   = test_df.dropna(subset=["binary_label"])

print(f"Train: {len(train_df)} rows | Valid: {len(valid_df)} rows | Test: {len(test_df)} rows")
print("Binary label distribution:\n", train_df["binary_label"].value_counts())

# ── 3. TF-IDF Vectorizer ──────────────────────────────────────────────────────
# TF-IDF converts text → numbers.
# TF = Term Frequency (how often a word appears in THIS statement)
# IDF = Inverse Document Frequency (penalizes words that appear in ALL statements)
# Result: rare but important words get high scores; common words like "the" get low scores.
# ngram_range=(1,2) means it also captures 2-word phrases like "fake news", "climate change"
# max_features=50000 means we keep the top 50,000 most important word/phrase features

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)

X_train = vectorizer.fit_transform(train_df["statement"])  # learns vocabulary from training data
X_valid = vectorizer.transform(valid_df["statement"])       # applies same vocab, doesn't relearn
X_test  = vectorizer.transform(test_df["statement"])

y_train = train_df["binary_label"]
y_valid = valid_df["binary_label"]
y_test  = test_df["binary_label"]

# ── 4. Train Logistic Regression ──────────────────────────────────────────────
# Logistic Regression draws a decision boundary in high-dimensional space.
# max_iter=1000 gives it enough steps to converge on 50k features.
# C=1.0 is regularization strength — prevents overfitting.

print("\nTraining Logistic Regression...")
model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
val_preds = model.predict(X_valid)
test_preds = model.predict(X_test)

print(f"\nValidation Accuracy: {accuracy_score(y_valid, val_preds):.4f}")
print(f"Test Accuracy:       {accuracy_score(y_test, test_preds):.4f}")
print("\nTest Classification Report:")
print(classification_report(y_test, test_preds))

# ── 6. Save model and vectorizer ──────────────────────────────────────────────
# We save both — the vectorizer must be saved too because it learned the vocabulary.
# When the app loads, it needs the SAME vectorizer to process new text the same way.

joblib.dump(model, "models/lr_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved to /models/")

# ── 7. Quick manual test ──────────────────────────────────────────────────────
test_statements = [
    "Scientists confirm vaccines are safe and effective",
    "Government secretly putting chips in drinking water",
    "New study links coffee to lower risk of diabetes",
    "Obama born in Kenya, documents prove it"
]

print("\n--- Manual Test ---")
for stmt in test_statements:
    vec = vectorizer.transform([stmt])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    confidence = max(prob) * 100
    print(f"  [{pred} {confidence:.1f}%] {stmt}")
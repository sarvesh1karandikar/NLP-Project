"""
train_and_save.py

Replicates the training logic from numericclassification.py, then saves
the best MLPClassifier to model.pkl and records metadata to model_meta.json.
"""

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from textblob import TextBlob
from textblob.classifiers import DecisionTreeClassifier

from score import get_score
from sentiment2 import feats

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(os.path.join(BASE_DIR, "totaldata.xlsx"))
X = df.drop(columns=["label", "team"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=42
)

X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Build TextBlob DecisionTree classifier from training text
# ---------------------------------------------------------------------------
train = pd.concat([X_train["text"], y_train], axis=1)
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train.to_csv(train_csv_path, index=False)

with open(train_csv_path, "r") as fp2:
    cl2 = DecisionTreeClassifier(fp2, format="csv")

# ---------------------------------------------------------------------------
# 3. Extract features for training set
# ---------------------------------------------------------------------------
pred_train   = []
feature_train = []
for instance in train["text"]:
    feature_train.append(feats(instance))
    blob = TextBlob(instance, classifier=cl2)
    pred_train.append(int(float(blob.classify())))

pred_train    = pd.DataFrame(pred_train)
feature_train = pd.DataFrame(feature_train)

closeness_train = pd.concat([X_train["score"], X_train["match_type"]], axis=1)
f_train = pd.Series(
    [get_score(row.iloc[0], row.iloc[1]) for _, row in closeness_train.iterrows()]
)

X_train_final = pd.concat([pred_train, feature_train, f_train], axis=1)

# ---------------------------------------------------------------------------
# 4. Extract features for test set
# ---------------------------------------------------------------------------
test = pd.concat([X_test["text"], y_test], axis=1)

pred_test    = []
feature_test = []
for instance in test["text"]:
    feature_test.append(feats(instance))
    blob = TextBlob(instance, classifier=cl2)
    pred_test.append(int(float(blob.classify())))

pred_test    = pd.DataFrame(pred_test)
feature_test = pd.DataFrame(feature_test)

closeness_test = pd.concat([X_test["score"], X_test["match_type"]], axis=1)
f_test = pd.Series(
    [get_score(row.iloc[0], row.iloc[1]) for _, row in closeness_test.iterrows()]
)

X_test_final = pd.concat([pred_test, feature_test, f_test], axis=1)

# ---------------------------------------------------------------------------
# 5. Grid search over MLP hidden layer sizes — same as numericclassification.py
# ---------------------------------------------------------------------------
accuracy_final = 0.0
clf_best = None

for m in range(4, 8):
    for n in range(1, 7):
        clf = MLPClassifier(
            solver="lbfgs",
            alpha=1e-5,
            warm_start=True,
            beta_1=0.95,
            hidden_layer_sizes=(m, n),
            random_state=1,
        )
        clf.fit(X_train_final, y_train)
        y_pred = clf.predict(X_test_final)

        count = sum(1 for i in range(len(y_pred)) if y_pred[i] == y_test[i])
        accuracy = count / len(y_pred)

        if accuracy > accuracy_final:
            accuracy_final = accuracy
            clf_best = clf

print(f"Best test accuracy: {accuracy_final:.4f}")

# ---------------------------------------------------------------------------
# 6. Save model and metadata
# ---------------------------------------------------------------------------
model_path = os.path.join(BASE_DIR, "model.pkl")
joblib.dump(clf_best, model_path)
print(f"Model saved to {model_path}")

meta = {
    "accuracy": round(accuracy_final, 4),
    "trained_at": datetime.now(timezone.utc).isoformat(),
}
meta_path = os.path.join(BASE_DIR, "model_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved to {meta_path}")

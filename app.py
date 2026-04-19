"""
app.py

Gradio demo for the NLP cricket excitement classifier.
Loads model.pkl at startup (trains it first if missing).
"""

import json
import os
import subprocess
import sys

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
from textblob.classifiers import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Lazy-import project modules (they live in the same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, BASE_DIR)
from score import get_score
from sentiment2 import feats

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
LABEL_MAP = {
    1: "Dull (1/5)",
    2: "Quiet (2/5)",
    3: "Decent (3/5)",
    4: "Thrilling (4/5)",
    5: "Edge-of-seat! (5/5)",
}

EXCITEMENT_ICONS = {1: "", 2: "⚡", 3: "⚡⚡", 4: "⚡⚡⚡", 5: "⚡⚡⚡⚡⚡"}


# ---------------------------------------------------------------------------
# Build TextBlob classifier (needed to replicate the training-time feature)
# ---------------------------------------------------------------------------
def _build_textblob_clf():
    """Rebuild the TextBlob DecisionTreeClassifier used during feature extraction."""
    train_csv = os.path.join(BASE_DIR, "train.csv")
    if not os.path.exists(train_csv):
        # Regenerate from the excel data
        import pandas as _pd
        from sklearn.model_selection import train_test_split as _tts

        df = _pd.read_excel(os.path.join(BASE_DIR, "totaldata.xlsx"))
        X = df.drop(columns=["label", "team"])
        y = df["label"]
        X_tr, _, y_tr, _ = _tts(X, y, test_size=0.3, shuffle=True, random_state=42)
        X_tr = X_tr.reset_index(drop=True)
        y_tr = y_tr.reset_index(drop=True)
        train_df = _pd.concat([X_tr["text"], y_tr], axis=1)
        train_df.to_csv(train_csv, index=False)

    with open(train_csv, "r") as fp:
        cl2 = DecisionTreeClassifier(fp, format="csv")
    return cl2


# ---------------------------------------------------------------------------
# Model loading / training
# ---------------------------------------------------------------------------
def _ensure_model():
    """Return the trained MLPClassifier, training it first if model.pkl is absent."""
    model_path = os.path.join(BASE_DIR, "model.pkl")
    if not os.path.exists(model_path):
        print("model.pkl not found — running train_and_save.py …")
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "train_and_save.py")],
            capture_output=True,
            text=True,
            cwd=BASE_DIR,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Training failed:\n{result.stdout}\n{result.stderr}"
            )
        print(result.stdout)
    return joblib.load(model_path)


# Load once at startup
print("Loading model …")
clf = _ensure_model()
cl2 = _build_textblob_clf()

meta_path = os.path.join(BASE_DIR, "model_meta.json")
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    model_info = f"Model accuracy: {meta['accuracy']:.1%}  |  Trained: {meta['trained_at'][:10]}"
else:
    model_info = ""

print("Model ready.")


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict_excitement(summary: str, margin: str, match_format: str) -> str:
    """Build the 7-feature vector and return a formatted excitement label."""
    summary = summary.strip()
    margin  = margin.strip()

    if not summary:
        return "Please enter a match summary."
    if not margin:
        return "Please enter the win margin (e.g. '5 runs' or '3 wickets')."

    # Feature 1: TextBlob DecisionTree sentiment prediction
    blob = TextBlob(summary, classifier=cl2)
    tb_pred = int(float(blob.classify()))

    # Features 2-6: NLP features from feats()
    nlp_feats = feats(summary)  # tuple of 5 values

    # Feature 7: closeness score from margin + format
    closeness = get_score(margin, match_format)
    if closeness is None:
        closeness = 5  # default mid-value if margin format not recognised

    # Assemble feature vector — same column order as training
    X = np.array([[tb_pred, *nlp_feats, closeness]], dtype=float)

    label = int(clf.predict(X)[0])
    label = max(1, min(5, label))  # clamp to valid range

    icon  = EXCITEMENT_ICONS.get(label, "")
    name  = LABEL_MAP.get(label, str(label))
    stars = "★" * label + "☆" * (5 - label)

    return f"{icon} {name}\n{stars}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
examples = [
    [
        (
            "India beat Pakistan in a nail-biting Super Over thriller. "
            "Virat Kohli smashed a brilliant 82 off 53 balls. "
            "The crowd went berserk as the last-ball six sealed the chase. "
            "An absolutely outstanding performance under pressure!"
        ),
        "2 runs",
        "T20",
    ],
    [
        (
            "Australia dominated England from start to finish. "
            "Warner hit a fluent century and the bowlers barely gave England a chance. "
            "England were bundled out for 180, losing by a comfortable margin."
        ),
        "95 runs",
        "ODI",
    ],
    [
        (
            "The Test match ended in a dramatic draw after a rain interruption on day five. "
            "The tail-enders batted heroically to deny the opposition a famous victory. "
            "Controversy erupted over a disputed DRS call that could have changed the result."
        ),
        "tie",
        "Test",
    ],
]

with gr.Blocks(title="Cricket Excitement Classifier") as demo:
    gr.Markdown(
        """
        # Cricket Match Excitement Classifier
        Rate how exciting a cricket match was on a **1–5 scale** using NLP + match context.
        """
    )
    if model_info:
        gr.Markdown(f"*{model_info}*")

    with gr.Row():
        with gr.Column(scale=2):
            summary_box = gr.Textbox(
                label="Match Summary",
                lines=6,
                placeholder=(
                    "Enter a short narrative of the match — "
                    "e.g. 'India won a thrilling last-ball finish against Australia…'"
                ),
            )
            margin_box = gr.Textbox(
                label="Win Margin",
                placeholder="e.g.  5 runs  |  3 wickets  |  tie",
            )
            format_drop = gr.Dropdown(
                label="Match Format",
                choices=["T20", "ODI", "Test"],
                value="T20",
            )
            predict_btn = gr.Button("Predict Excitement", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="Excitement Rating",
                lines=3,
                interactive=False,
            )

    predict_btn.click(
        fn=predict_excitement,
        inputs=[summary_box, margin_box, format_drop],
        outputs=output_box,
    )

    gr.Examples(
        examples=examples,
        inputs=[summary_box, margin_box, format_drop],
        outputs=output_box,
        fn=predict_excitement,
        cache_examples=False,
        label="Try an example",
    )

if __name__ == "__main__":
    demo.launch()

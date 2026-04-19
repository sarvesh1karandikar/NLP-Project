# CLAUDE.md — Cricket Match Excitement Classifier

## Project Summary

Multi-class NLP classifier that takes a cricket match summary (plain English prose, ~100-300 words) and outputs an excitement rating on a 1-5 ordinal scale. Built as a USC NLP course project. Uses a hand-crafted feature pipeline (sentence sentiment + domain lexicons + structured match metadata) feeding an MLP neural network.

**Unique angle:** The cricket domain makes this immediately memorable in a portfolio — the demo is concrete ("paste commentary, get a score 1-5") and the domain-specific lexicon engineering shows deliberate NLP thinking beyond off-the-shelf models.

---

## How to Run

Prerequisites: Python 3.7+, all 6 source files in the same directory.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
python numericclassification.py
```

The script prints `accuracy_final` — the best test accuracy found across the MLP grid search.

File dependencies:
- `totaldata.xlsx` — main dataset (text + structured features)
- `train_sent_final.json` — training data for TextBlob NaiveBayesClassifier
- `bagofwords_dict.py` — WordNet lexicon expansion
- `sentiment2.py` — feature extraction (sentence sentiment + lexicon counts)
- `score.py` — match closeness score from structured margin column
- `numericclassification.py` — main training/evaluation script

---

## Model Details

**Feature vector (per match summary):**

| Feature | Source | Description |
|---------|--------|-------------|
| TextBlob label | `sentiment2.py` | DecisionTreeClassifier prediction (int 1-5) |
| pos/neg ratio | `sentiment2.py` | ratio of positive to negative sentences |
| excitement_count | `sentiment2.py` | count of WordNet-expanded excitement words |
| controversy_count | `sentiment2.py` | count of WordNet-expanded controversy words |
| rain_count | `sentiment2.py` | count of rain-related words (rain = major match event) |
| up_down_count | `sentiment2.py` | count of turn-of-play markers ("but", "although") |
| closeness_score | `score.py` | integer derived from win margin + format (T20 vs ODI) |

**Classifier:** `sklearn.neural_network.MLPClassifier`
- solver: `lbfgs`
- alpha: `1e-5`
- beta_1: `0.95`
- warm_start: True
- Grid search over hidden_layer_sizes `(m, n)` where `m in range(4,8)`, `n in range(1,7)` — 24 combinations

**Train/test split:** 70/30, shuffled, no fixed random seed (accuracy varies per run)

---

## Current Accuracy and Limitations

- **Reported accuracy:** ~55-70% on 5-class ordinal classification (random baseline = 20%)
- The grid search selects best MLP config by test accuracy, which can overfit to the specific split

**Limitations:**
1. **Small dataset (~200 samples):** Insufficient for deep learning; MLP is near the ceiling for this data size
2. **No fixed random seed:** Results are not reproducible run-to-run
3. **Text features are bag-of-words-adjacent:** No word order, no named entity awareness (player names, teams carry strong excitement signal)
4. **Structured features require a clean `score` column:** The model cannot classify from raw text alone without the margin/format metadata
5. **Class imbalance:** Labels 1 and 5 are rarer; model likely underperforms on extremes
6. **No confusion matrix logged:** Hard to diagnose which confusions are harmless (4 vs 5) vs damaging (1 vs 5)

---

## Enhancement TODO

### Quick Win
- [ ] Add a fixed `random_state` to `train_test_split` for reproducibility
- [ ] Log confusion matrix and per-class F1 scores (use `sklearn.metrics.classification_report`)
- [ ] Add a simple inference function: `predict_excitement(summary_text) -> int`
- [ ] Replace hard-coded `train.csv` write with an in-memory approach to avoid file I/O bugs

### Medium Lift
- [ ] **Gradio UI:** Wrap inference in a Gradio interface — paste commentary, get score 1-5 with a label like "Classic Thriller". Deployable on Hugging Face Spaces for free
- [ ] **Expand dataset:** Scrape Cricinfo match reports via their undocumented API or use the existing Cricsheet data + manual labels. Target: 1000+ samples
- [ ] **Add sentiment per player:** Named entity recognition (spaCy) to track sentiment around individual player mentions — a strong signal (Kohli praised = India doing well)
- [ ] **IPL commentary:** Extend to IPL domestic T20s — larger corpus, well-archived, huge portfolio appeal for Indian tech recruiters

### Big Lift
- [ ] **Fine-tune BERT / DistilBERT:** Replace the entire feature pipeline with a pre-trained transformer fine-tuned on the labelled corpus. With ~1000 samples DistilBERT should outperform the MLP significantly. Use `transformers` + `Trainer` API
- [ ] **Ordinal regression loss:** Replace standard cross-entropy with an ordinal loss (e.g., `coral-pytorch`) to penalise distant-class errors less
- [ ] **Multi-modal model:** Combine text summary + ball-by-ball scoring data (runs per over, wickets per phase) as structured features alongside BERT embeddings
- [ ] **Live demo pipeline:** Connect to the Cricinfo live commentary feed, classify excitement in real time, and push to a public leaderboard

---

## Recommended Demo Tier

**Recommended: Medium Lift — Gradio UI on Hugging Face Spaces**

**Justification:** A Gradio interface requires only adding ~20 lines of code around the existing `predict_excitement` function and deploys for free on Hugging Face Spaces. The cricket domain is instantly engaging for any recruiter or technical interviewer — they can paste a famous match summary (e.g., the 2011 World Cup final) and see the model output 5, which creates a memorable "it works!" moment that raw accuracy numbers never achieve.

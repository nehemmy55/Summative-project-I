"""Metrics: MCQ accuracy, exact match, token F1, ROUGE and bootstrap confidence intervals."""
import re
import string
from collections import Counter

import numpy as np
from rouge_score import rouge_scorer

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({ch: " " for ch in string.punctuation})
# matches "C", "(D)", "A. Iron" but not "A patient ..."
_LETTER = re.compile(r"^(?:([A-Ha-h])$|\(([A-Ha-h])\)|([A-Ha-h])[\.\):](?:\s|$))")
_rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def normalise(text):
    """Lower-case, remove punctuation and articles (SQuAD answer normalisation)."""
    text = str(text).lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalise(prediction) == normalise(reference))


def token_f1(prediction, reference):
    pred, ref = normalise(prediction).split(), normalise(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def pick_option(prediction, options):
    """Index of the option closest to the generated text, or -1 if nothing overlaps."""
    raw_pred = str(prediction).strip().lower()
    norm_pred = normalise(prediction)
    for match in (lambda o: o.strip().lower() == raw_pred, lambda o: normalise(o) == norm_pred):
        hits = [i for i, option in enumerate(options) if match(option)]
        if len(hits) == 1:
            return hits[0]
    # the model sometimes answers with just the letter
    letter = _LETTER.match(str(prediction).strip())
    if letter:
        index = "ABCDEFGH".index((letter.group(1) or letter.group(2) or letter.group(3)).upper())
        if index < len(options):
            return index
    scores = [token_f1(prediction, option) for option in options]
    best = int(np.argmax(scores))
    # tie between options = no clear choice
    if scores[best] == 0 or scores.count(scores[best]) > 1:
        return -1
    return best


def bootstrap_ci(values, n_boot=2000, seed=0):
    """95% confidence interval of the mean by resampling the test examples."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def score_rows(df):
    """Add per-example scores to a frame with columns task, answer, options, prediction."""
    df = df.copy()
    df["prediction"] = df.prediction.fillna("").astype(str)
    df["em"] = [exact_match(p, a) for p, a in zip(df.prediction, df.answer)]
    df["f1"] = [token_f1(p, a) for p, a in zip(df.prediction, df.answer)]
    rouge = [_rouge.score(a, p) for p, a in zip(df.prediction, df.answer)]
    df["rouge1"] = [r["rouge1"].fmeasure for r in rouge]
    df["rougeL"] = [r["rougeL"].fmeasure for r in rouge]
    picked, correct = [], []
    for row in df.itertuples():
        if row.task == "mcq":
            options = list(row.options)
            idx = pick_option(row.prediction, options)
            picked.append(options[idx] if idx >= 0 else "")
            correct.append(float(idx >= 0 and options[idx] == row.answer))
        else:
            picked.append("")
            correct.append(np.nan)
    df["picked_option"], df["correct"] = picked, correct
    return df


def summarise(scored, with_bertscore=False):
    """Aggregate per-example scores into the numbers reported in the results table."""
    out = {}
    mcq = scored[scored.task == "mcq"]
    if len(mcq):
        out["mcq"] = {
            "n": len(mcq),
            "accuracy": round(mcq.correct.mean(), 4),
            "accuracy_ci95": bootstrap_ci(mcq.correct),
            "random_baseline": round(float(np.mean([1 / len(o) for o in mcq.options])), 4),
            "no_option_matched": round(float((mcq.picked_option == "").mean()), 4),
        }
    subsets = {
        "open_afrimed": scored[(scored.task == "open") & (scored.source == "afrimedqa")],
        "open_medquad": scored[(scored.task == "open") & (scored.source == "medquad")],
    }
    for name, part in subsets.items():
        if not len(part):
            continue
        out[name] = {
            "n": len(part),
            "exact_match": round(part.em.mean(), 4),
            "token_f1": round(part.f1.mean(), 4),
            "rouge1": round(part.rouge1.mean(), 4),
            "rougeL": round(part.rougeL.mean(), 4),
            "rougeL_ci95": bootstrap_ci(part.rougeL),
        }
        if with_bertscore:
            from bert_score import score as bert_score
            _, _, f = bert_score(part.prediction.tolist(), part.answer.tolist(),
                                 model_type="distilbert-base-uncased", verbose=False)
            out[name]["bertscore_f1"] = round(float(f.mean()), 4)
    return out

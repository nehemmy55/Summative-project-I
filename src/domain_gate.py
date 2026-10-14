"""Health vs not-health question classifier (the domain gate)."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

from . import config as C
from .retrieval import TfidfRetriever

GATE_PATH = C.MODEL_DIR / "domain_gate.joblib"

# a few questions I wrote by hand to check greetings and other topics
PROBES = [
    ("hello", 0), ("how are you today?", 0), ("who won the africa cup of nations in 2019?", 0),
    ("what is the capital of rwanda?", 0), ("how do I fix a python import error?", 0),
    ("what is the best fertilizer for maize?", 0), ("how can I open a bank account?", 0),
    ("write me a poem about love", 0), ("what is the exchange rate of the dollar?", 0),
    ("how many players are in a football team?", 0),
    ("what are the symptoms of malaria?", 1), ("how is typhoid fever treated?", 1),
    ("my child has diarrhoea and is vomiting, what should I do?", 1),
    ("is it safe to take paracetamol while pregnant?", 1), ("how can I prevent cholera?", 1),
    ("what causes sickle cell anaemia?", 1), ("i have a headache that will not go away", 1),
    ("how is HIV transmitted from mother to child?", 1), ("what is hypertension?", 1),
    ("can tuberculosis be cured?", 1),
]


def report(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "confusion_matrix[[TN,FP],[FN,TP]]": confusion_matrix(y_true, y_pred).tolist()}


def load_gate():
    # only load the file made by main() below (joblib can run code on load)
    return joblib.load(GATE_PATH)


def is_health_question(gate, text, threshold=0.5):
    probability = float(gate.predict_proba([text])[0, 1])
    return probability >= threshold, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true")
    args = parser.parse_args()

    train = pd.read_json(C.PROC_DIR / "ood_train.jsonl", lines=True)
    test = pd.read_json(C.PROC_DIR / "ood_test.jsonl", lines=True)
    probe_text, probe_label = [p[0] for p in PROBES], np.array([p[1] for p in PROBES])
    results = {"train_size": len(train), "test_size": len(test)}

    # 1) threshold on retrieval similarity
    retriever = TfidfRetriever()
    tune = train.sample(3000, random_state=C.SEED)
    tune_scores = np.array([h["score"] for h in retriever.search(tune.text.tolist(), exclude_ids=None)])
    thresholds = np.arange(0.05, 0.9, 0.05)
    best_t = max(thresholds, key=lambda t: accuracy_score(tune.label, tune_scores >= t))
    test_scores = np.array([h["score"] for h in retriever.search(test.text.tolist())])
    probe_scores = np.array([h["score"] for h in retriever.search(probe_text)])
    results["similarity_threshold"] = {"threshold": round(float(best_t), 2),
                                       **report(test.label, test_scores >= best_t),
                                       "probe_accuracy": round(accuracy_score(probe_label, probe_scores >= best_t), 2)}

    # 2) char n-gram TF-IDF + logistic regression
    gate = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced"),
    )
    gate.fit(train.text, train.label)
    pred = gate.predict(test.text)
    results["tfidf_logreg"] = {**report(test.label, pred),
                               "probe_accuracy": round(accuracy_score(probe_label, gate.predict(probe_text)), 2)}
    joblib.dump(gate, GATE_PATH, compress=3)

    errors = test[pred != test.label].assign(predicted=pred[pred != test.label])
    errors.to_csv(C.RESULTS_DIR / "domain_gate_errors.csv", index=False)
    probes = pd.DataFrame({"text": probe_text, "label": probe_label,
                           "p_health": gate.predict_proba(probe_text)[:, 1].round(3)})
    probes.to_csv(C.RESULTS_DIR / "domain_gate_probes.csv", index=False)

    # 3) MiniLM embeddings + logistic regression
    if args.embeddings:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        sub = train.sample(6000, random_state=C.SEED)   # subsample, encoding is slow on CPU
        clf = LogisticRegression(max_iter=1000).fit(encoder.encode(sub.text.tolist(), batch_size=64), sub.label)
        results["minilm_logreg"] = {
            "train_subsample": len(sub),
            **report(test.label, clf.predict(encoder.encode(test.text.tolist(), batch_size=64))),
            "probe_accuracy": round(accuracy_score(probe_label, clf.predict(encoder.encode(probe_text))), 2)}

    (C.RESULTS_DIR / "domain_gate_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(probes.to_string())


if __name__ == "__main__":
    main()

"""Runs a baseline or a trained model on the test set and saves predictions and metrics."""
import argparse
import json
import random
import time
from pathlib import Path

from . import config as C
from .metrics import score_rows, summarise
from .qa_model import add_prompts, generate, load_model, load_split
from .retrieval import TfidfRetriever


def predict(system, test):
    if system == "random":
        rng = random.Random(C.SEED)
        return [rng.choice(list(o)) if len(o) else "" for o in test.options], {}
    if system == "tfidf":
        hits = TfidfRetriever().search(test.question.tolist())
        return [h["kb_answer"] for h in hits], {"retrieval_score": [h["score"] for h in hits]}

    model, tokenizer, run_config = load_model(C.BASE_MODEL if system == "zeroshot" else system)
    use_rag = bool(run_config.get("rag"))
    test = add_prompts(test, TfidfRetriever() if use_rag else None)
    max_len = C.MAX_SOURCE_LEN_RAG if use_rag else C.MAX_SOURCE_LEN
    return generate(model, tokenizer, test.prompt.tolist(), max_len, progress=True), {}


def run(system, name=None, split="test", bertscore=False, limit=None):
    name = name or Path(system).name
    test = load_split(split)
    if limit:
        test = test.sample(limit, random_state=C.SEED).reset_index(drop=True)
    start = time.time()
    predictions, extra = predict(system, test)
    test["prediction"] = predictions
    for key, values in extra.items():
        test[key] = values
    scored = score_rows(test)
    metrics = summarise(scored, with_bertscore=bertscore)
    metrics["seconds"] = round(time.time() - start, 1)

    out_dir = C.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.drop(columns=["rationale"]).to_csv(out_dir / f"predictions_{split}.csv", index=False)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
    print(name, json.dumps(metrics, indent=2))
    return metrics


def rescore(name, split="test", bertscore=False):
    """Recompute the metrics of a finished run from its saved predictions."""
    import ast
    import pandas as pd
    out_dir = C.RESULTS_DIR / name
    saved = pd.read_csv(out_dir / f"predictions_{split}.csv", dtype={"id": str}, keep_default_na=False)
    saved["options"] = saved.options.map(ast.literal_eval)
    old = json.loads((out_dir / f"metrics_{split}.json").read_text())
    scored = score_rows(saved)
    metrics = summarise(scored, with_bertscore=bertscore)
    metrics["seconds"] = old.get("seconds")
    scored.to_csv(out_dir / f"predictions_{split}.csv", index=False)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", help="random | tfidf | zeroshot | path to a model directory")
    parser.add_argument("--rescore", help="name of a finished run to re-score from saved predictions")
    parser.add_argument("--name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.rescore:
        print(args.rescore, json.dumps(rescore(args.rescore, args.split, args.bertscore), indent=2))
    else:
        run(args.system, args.name, args.split, args.bertscore, args.limit)

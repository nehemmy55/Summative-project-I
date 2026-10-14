"""Makes the results table, plots and error analysis from the saved predictions."""
import argparse
import ast
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .metrics import normalise

FIG_DIR = C.RESULTS_DIR / "figures"
ORDER = ["random", "tfidf", "zeroshot", "lora_r8", "lora_r32", "full_ft", "full_ft_medquad",
         "full_ft_rag", "base_lora_r32", "base_full", "base_full_rag"]
LABELS = {
    "random": "Random option",
    "tfidf": "TF-IDF retrieval (baseline)",
    "zeroshot": "FLAN-T5-small zero-shot (baseline)",
    "lora_r8": "Small, LoRA r=8, AfriMed",
    "lora_r32": "Small, LoRA r=32, AfriMed",
    "full_ft": "Small, full FT, AfriMed",
    "full_ft_medquad": "Small, full FT, + MedQuAD",
    "full_ft_rag": "Small, full FT, + MedQuAD + retrieval",
    "base_lora_r32": "Base, LoRA r=32, AfriMed",
    "base_full": "Base, full FT, AfriMed",
    "base_full_rag": "Base, full FT, + MedQuAD + retrieval",
}
NEGATION = re.compile(r"\b(except|not|false|incorrect|least|never|untrue|wrong)\b", re.I)


def load_predictions(name):
    df = pd.read_csv(C.RESULTS_DIR / name / "predictions_test.csv", dtype={"id": str}, keep_default_na=False)
    df["options"] = df.options.map(ast.literal_eval)
    # keep_default_na: otherwise pandas reads the answer "None" as NaN
    for column in ["em", "f1", "rouge1", "rougeL", "correct"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def summary_table():
    rows = []
    for name in ORDER:
        path = C.RESULTS_DIR / name / "metrics_test.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        cfg_path = C.RESULTS_DIR / name / "run_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        ci = m["mcq"]["accuracy_ci95"]
        rows.append({
            "System": LABELS[name],
            "Trainable params": f"{cfg['trainable_params']:,}" if cfg else "-",
            "Train min": f"{cfg['train_minutes']} ({cfg.get('device', 'cpu')})" if cfg else "-",
            "Val loss": cfg.get("best_val_loss", "-"),
            "MCQ acc": f"{100 * m['mcq']['accuracy']:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]",
            "SAQ F1": f"{100 * m['open_afrimed']['token_f1']:.1f}",
            "SAQ ROUGE-L": f"{100 * m['open_afrimed']['rougeL']:.1f}",
            "MedQuAD F1": f"{100 * m['open_medquad']['token_f1']:.1f}",
            "MedQuAD ROUGE-L": f"{100 * m['open_medquad']['rougeL']:.1f}",
            "_name": name, "_acc": m["mcq"]["accuracy"], "_ci": ci,
            "_saq": m["open_afrimed"]["rougeL"], "_mq": m["open_medquad"]["rougeL"],
        })
    return pd.DataFrame(rows)


def plot_comparison(table):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    y = np.arange(len(table))[::-1]
    err = [[a - c[0] for a, c in zip(table._acc, table._ci)], [c[1] - a for a, c in zip(table._acc, table._ci)]]
    axes[0].barh(y, 100 * table._acc, xerr=100 * np.array(err), color="#3b6ea5", capsize=3)
    axes[0].axvline(20, color="grey", ls="--", lw=1)
    axes[0].text(20.3, len(table) - 0.45, "chance", color="grey", fontsize=8)
    axes[0].set_yticks(y, table.System)
    axes[0].set_xlabel("MCQ accuracy on the test set (%), 95% bootstrap CI")
    axes[1].barh(y + 0.2, 100 * table._saq, height=0.4, color="#3b6ea5", label="AfriMed-QA short answer")
    axes[1].barh(y - 0.2, 100 * table._mq, height=0.4, color="#c9793a", label="MedQuAD")
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].set_xlabel("ROUGE-L on open questions (%)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_comparison.png", dpi=160)
    plt.close(fig)


def plot_loss_curves():
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for name in ORDER:
        path = C.RESULTS_DIR / name / "run_config.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text())["history"]
        epochs = [h["epoch"] for h in history]
        line, = ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label=LABELS[name])
        ax.plot(epochs, [h["train_loss"] for h in history], ls="--", color=line.get_color(), alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Solid: validation, dashed: training", fontsize=9)
    ax.set_xticks(epochs)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loss_curves.png", dpi=160)
    plt.close(fig)


def open_error_type(row):
    """Rough automatic label for an open answer, used to count error types."""
    pred, question = normalise(row.prediction).split(), set(normalise(row.question).split())
    if not pred:
        return "empty"
    if row.rougeL >= 0.5:
        return "good (ROUGE-L >= 0.5)"
    if len(pred) >= 6 and len(set(pred)) / len(pred) < 0.6:
        return "repetitive"
    if len(pred) >= 4 and sum(w in question for w in pred) / len(pred) >= 0.7:
        return "restates the question"
    if row.rougeL >= 0.25:
        return "partially correct"
    return "wrong or unrelated content"


def error_analysis(name):
    df = load_predictions(name)
    mcq = df[df.task == "mcq"].copy()
    lines = [f"# Error analysis for `{name}` ({LABELS.get(name, name)})\n"]

    # MCQ part
    mcq["negated"] = mcq.question.str.contains(NEGATION)
    mcq["q_words"] = pd.cut(mcq.question.str.split().str.len(), [0, 10, 25, 1000],
                            labels=["short (<=10 words)", "medium (11-25)", "long (>25, clinical vignette)"])
    mcq["gold_pos"] = [o.index(a) for o, a in zip(mcq.options, mcq.answer)]
    mcq["pred_pos"] = [o.index(p) if p in o else -1 for o, p in zip(mcq.options, mcq.picked_option.fillna(""))]
    mcq["answer_words"] = pd.cut(mcq.answer.str.split().str.len(), [0, 2, 5, 100],
                                 labels=["1-2 words", "3-5 words", "6+ words"])

    def breakdown(column, title):
        table = mcq.groupby(column, observed=True).correct.agg(n="size", accuracy="mean")
        table["accuracy"] = (100 * table.accuracy).round(1)
        lines.append(f"\n**MCQ accuracy by {title}**\n\n" + table.to_markdown() + "\n")
        return table

    lines.append(f"## Multiple-choice questions (n={len(mcq)}, accuracy {100 * mcq.correct.mean():.1f}%)\n")
    breakdown("negated", "negation in the question (except / not / false ...)")
    breakdown("q_words", "question length")
    breakdown("answer_words", "length of the correct option")
    breakdown("country", "country of the contributor")
    top = mcq.specialty.value_counts().head(8).index
    spec = mcq[mcq.specialty.isin(top)].groupby("specialty").correct.agg(n="size", accuracy="mean")
    spec["accuracy"] = (100 * spec.accuracy).round(1)
    spec = spec.sort_values("accuracy")
    lines.append("\n**MCQ accuracy by specialty (8 largest)**\n\n" + spec.to_markdown() + "\n")

    positions = pd.DataFrame({
        "correct option (%)": (100 * mcq.gold_pos.value_counts(normalize=True)).round(1),
        "model choice (%)": (100 * mcq.pred_pos.value_counts(normalize=True)).round(1),
    }).fillna(0).sort_index()
    positions.index = ["no match" if i == -1 else "ABCDEFGH"[i] for i in positions.index]
    lines.append("\n**Position of the chosen option**\n\n" + positions.to_markdown() + "\n")
    all_above = mcq.options.map(lambda o: any("all of the above" in x.lower() or "none of the above" in x.lower() for x in o))
    lines.append(f"\nQuestions with an 'all/none of the above' option: n={int(all_above.sum())}, "
                 f"accuracy {100 * mcq[all_above].correct.mean():.1f}%\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].barh(spec.index.str.replace("_", " "), spec.accuracy, color="#3b6ea5")
    axes[0].axvline(20, color="grey", ls="--", lw=1)
    axes[0].set_xlabel("MCQ accuracy (%)")
    positions.plot.bar(ax=axes[1], color=["#9aa5b1", "#3b6ea5"], rot=0)
    axes[1].set_ylabel("% of test questions")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mcq_error_analysis.png", dpi=160)
    plt.close(fig)

    def examples(frame, columns, n):
        out = []
        for row in frame.head(n).itertuples():
            out.append("\n".join(f"- **{c}:** {getattr(row, c)}" for c in columns) + "\n")
        return "\n".join(out)

    lines.append("\n### Correct MCQ examples\n")
    lines.append(examples(mcq[mcq.correct == 1].sample(frac=1, random_state=1),
                          ["question", "options", "answer", "prediction"], 5))
    lines.append("\n### Incorrect MCQ examples\n")
    lines.append(examples(mcq[mcq.correct == 0].sample(frac=1, random_state=1),
                          ["question", "options", "answer", "prediction", "specialty"], 8))

    # open questions
    open_q = df[df.task == "open"].copy()
    open_q["error_type"] = open_q.apply(open_error_type, axis=1)
    counts = open_q.groupby(["source", "error_type"]).size().unstack(0).fillna(0).astype(int)
    lines.append(f"\n## Open questions (n={len(open_q)})\n\n**Automatic error categories**\n\n" + counts.to_markdown() + "\n")
    lines.append("\n### Best open answers\n")
    lines.append(examples(open_q.sort_values("rougeL", ascending=False),
                          ["source", "question", "answer", "prediction", "rougeL"], 5))
    for kind in ["wrong or unrelated content", "restates the question", "repetitive", "partially correct"]:
        part = open_q[open_q.error_type == kind]
        if len(part):
            lines.append(f"\n### Open answers: {kind}\n")
            lines.append(examples(part.sample(frac=1, random_state=2),
                                  ["source", "question", "answer", "prediction", "rougeL"], 4))

    open_q.to_csv(C.RESULTS_DIR / name / "open_error_types.csv", index=False)
    (C.RESULTS_DIR / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def possible_causes(row):
    """Properties of the example that the breakdown tables link to more errors (hints, not proof)."""
    causes = []
    if row.similarity < 0.2:
        causes.append("nothing similar in the knowledge base")
    if any(ch in row.answer for ch in "<{}"):
        causes.append("answer has a symbol the T5 tokenizer cannot produce")
    if row.task == "mcq":
        if len(row.answer.split()) >= 6:
            causes.append("long correct option")
        if NEGATION.search(row.question):
            causes.append("negated question")
        if any("of the above" in option.lower() for option in row.options):
            causes.append("'all/none of the above' option")
        if row.picked_option and list(row.options).index(row.picked_option) == 2:
            causes.append("picked option C (position bias)")
    elif re.match(r"(?i)\s*(discuss|describe|write|outline|explain|list|classify|name)\b", row.question):
        causes.append("instruction needs a long structured answer")
    return "; ".join(causes)


def error_table(final):
    """Input, expected, predicted, correct or not, error type and possible causes for every test example."""
    from .qa_model import load_split
    from .retrieval import TfidfRetriever

    df = load_predictions(final)
    df["similarity"] = [h["score"] for h in TfidfRetriever().search(load_split("test").question.tolist())]
    df["picked_option"] = df.picked_option.fillna("")
    is_mcq = df.task == "mcq"
    df["error_type"] = df.apply(open_error_type, axis=1)
    df.loc[is_mcq, "error_type"] = [
        "correct" if c == 1 else ("no option matched" if not p else "wrong option")
        for c, p in zip(df[is_mcq].correct, df[is_mcq].picked_option)]
    df["is_correct"] = [bool(c == 1) if t == "mcq" else bool(r >= 0.5)
                        for c, t, r in zip(df.correct, df.task, df.rougeL)]
    df["possible_cause"] = [possible_causes(row) if not ok else "" for row, ok in zip(df.itertuples(), df.is_correct)]
    table = df.rename(columns={"question": "input", "answer": "expected", "prediction": "predicted"})[
        ["source", "task", "input", "options", "expected", "predicted", "is_correct", "error_type",
         "possible_cause", "rougeL", "similarity"]]
    table.to_csv(C.RESULTS_DIR / "error_table.csv", index=False)

    wrong = df[~df.is_correct]
    counts = wrong.possible_cause.str.split("; ").explode()
    counts = counts[counts != ""].value_counts()
    with open(C.RESULTS_DIR / "error_analysis.md", "a", encoding="utf-8") as handle:
        handle.write("\n\n## Possible causes attached to the wrong answers\n\n"
                     f"{len(wrong)} of {len(df)} test answers are wrong (MCQ: wrong option; open: ROUGE-L < 0.5). "
                     "One answer can have several tags or none; the tags are rule-based hints. "
                     "The full table is in `results/error_table.csv`.\n\n"
                     + counts.rename("wrong answers").to_frame().to_markdown() + "\n")


def chance_and_retrieval_analysis(final, no_retrieval="full_ft_medquad"):
    """Two follow-up questions: is MCQ accuracy above chance, and when does retrieval help?"""
    from scipy.stats import binomtest

    from .metrics import token_f1
    from .qa_model import load_split
    from .retrieval import TfidfRetriever

    lines = ["\n## Is MCQ accuracy above chance?\n",
             "One-sided binomial test against 20% (five options) on the 373 test MCQs.\n",
             "| System | correct / n | accuracy | p-value |", "|---|---|---|---|"]
    correct = {}
    for name in ORDER:
        if not (C.RESULTS_DIR / name / "predictions_test.csv").exists() or name in ("random", "tfidf"):
            continue
        c = load_predictions(name)
        correct[name] = c[c.task == "mcq"].correct.reset_index(drop=True)
        k, n = int(correct[name].sum()), len(correct[name])
        p_value = binomtest(k, n, 0.2, alternative="greater").pvalue
        lines.append(f"| {LABELS[name]} | {k} / {n} | {100 * k / n:.1f}% | {p_value:.3f} |")
    tuned = pd.concat([v for k, v in correct.items() if k != "zeroshot"], axis=1)
    n_right = tuned.sum(axis=1)
    lines.append(f"\nAcross the {tuned.shape[1]} fine-tuned models, {100 * (n_right == 0).mean():.1f}% of the "
                 f"questions are answered correctly by none and {100 * (n_right == tuned.shape[1]).mean():.1f}% by all; "
                 f"if the models were guessing independently these would be about "
                 f"{100 * 0.8 ** tuned.shape[1]:.1f}% and {100 * 0.2 ** tuned.shape[1]:.3f}%.\n")

    # does retrieval help more when the retrieved entry is similar?
    test = load_split("test")
    hits = TfidfRetriever().search(test.question.tolist())
    rag, plain, tfidf = load_predictions(final), load_predictions(no_retrieval), load_predictions("tfidf")
    frame = pd.DataFrame({
        "source": test.source, "task": test.task, "similarity": [h["score"] for h in hits],
        "with retrieval": rag.rougeL.values, "without retrieval": plain.rougeL.values,
        "TF-IDF answer": tfidf.rougeL.values,
        "context/answer F1": [token_f1(h["kb_passage"], a) for h, a in zip(hits, test.answer)],
        "output copies context": [token_f1(p, h["kb_passage"]) >= 0.8 for p, h in zip(rag.prediction, hits)],
    })
    open_q = frame[frame.task == "open"].copy()
    open_q["similarity bin"] = pd.cut(open_q.similarity, [0, 0.2, 0.4, 0.6, 1.01],
                                      labels=["< 0.2", "0.2-0.4", "0.4-0.6", ">= 0.6"])
    table = open_q.groupby(["source", "similarity bin"], observed=True).agg(
        n=("similarity", "size"), **{c: (c, "mean") for c in
                                     ["with retrieval", "without retrieval", "TF-IDF answer",
                                      "context/answer F1", "output copies context"]})
    lines.append("\n## When does retrieval help? (open questions, ROUGE-L by TF-IDF similarity of the retrieved entry)\n")
    lines.append((100 * table.drop(columns="n")).round(1).assign(n=table.n).to_markdown() + "\n")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for column, color in [("with retrieval", "#3b6ea5"), ("without retrieval", "#c9793a"), ("TF-IDF answer", "#9aa5b1")]:
        means = open_q.groupby("similarity bin", observed=True)[column].mean()
        ax.plot(means.index.astype(str), 100 * means.values, marker="o", color=color, label=column)
    ax.set_xlabel("TF-IDF similarity of the retrieved knowledge-base entry")
    ax.set_ylabel("ROUGE-L on open questions (%)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "retrieval_effect.png", dpi=160)
    plt.close(fig)

    with open(C.RESULTS_DIR / "error_analysis.md", "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", default="full_ft_rag", help="run whose errors are analysed")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    table = summary_table()
    plot_comparison(table)
    plot_loss_curves()
    public = table[[c for c in table.columns if not c.startswith("_")]]
    (C.RESULTS_DIR / "summary.md").write_text(public.to_markdown(index=False), encoding="utf-8")
    print(public.to_string(index=False))
    error_analysis(args.final)
    chance_and_retrieval_analysis(args.final)
    error_table(args.final)
    print(f"\nwrote results/summary.md, results/error_analysis.md and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()

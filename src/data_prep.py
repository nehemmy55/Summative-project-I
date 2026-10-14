"""Downloads the data, cleans it and builds the train/val/test files."""
import json
import re
import urllib.request

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from . import config as C


def clean_text(text):
    """Collapse whitespace and line breaks; drop characters broken by a bad encoding."""
    text = str(text).replace("\ufffd", " ")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_question(text):
    text = clean_text(text)
    # remove question numbers like "31. "
    text = re.sub(r"^[\.\s]*\d{0,3}[\.\)]\s+", "", text).lstrip(". ")
    return text


def clean_option(text, index):
    """Remove labels like "a." or "B)", only if the letter matches the position (so "E. coli" stays)."""
    text = clean_text(text)
    letter = "abcdefgh"[index] if index < 8 else "?"
    text = re.sub(rf"^[\(\[]?{letter}[\.\)\]:]\s*", "", text, flags=re.I)
    return text.rstrip(". ").strip()


def shorten(text, limit=C.ANSWER_WORD_LIMIT):
    """Keep whole sentences until the word limit is reached (at least one sentence)."""
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    kept, n_words = [], 0
    for sent in sentences:
        words = len(sent.split())
        if kept and n_words + words > limit:
            break
        kept.append(sent)
        n_words += words
    return " ".join(" ".join(kept).split()[: limit + 15])


def normalise_key(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def load_afrimed():
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not C.AFRIMED_CSV.exists():
        print("downloading AfriMed-QA ...")
        urllib.request.urlretrieve(C.AFRIMED_URL, C.AFRIMED_CSV)
    return pd.read_csv(C.AFRIMED_CSV, low_memory=False)


def build_afrimed_qa(df):
    """Turn the answered AfriMed-QA rows (MCQ and SAQ) into question-answer records."""
    records, dropped = [], {"mcq_multi_answer": 0, "mcq_bad_options": 0, "empty": 0}
    for row in df.itertuples():
        question = clean_question(row.question_clean)
        base = {
            "id": row.sample_id[:12],
            "source": "afrimedqa",
            "country": row.country,
            "specialty": row.specialty if isinstance(row.specialty, str) else "Unknown",
            "tier": row.tier,
            "question": question,
        }
        if row.question_type == "mcq":
            # skip questions with more than one correct option
            if "," in str(row.correct_answer):
                dropped["mcq_multi_answer"] += 1
                continue
            try:
                raw = json.loads(row.answer_options)
            except (TypeError, ValueError):
                dropped["mcq_bad_options"] += 1
                continue
            keys = [k for k in sorted(raw) if raw[k] is not None and clean_text(raw[k])]
            options = [clean_option(raw[k], i) for i, k in enumerate(keys)]
            if row.correct_answer not in keys or len(options) < 2 or not question or not all(options):
                dropped["mcq_bad_options"] += 1
                continue
            answer = options[keys.index(row.correct_answer)]
            rationale = row.answer_rationale if isinstance(row.answer_rationale, str) else ""
            records.append({**base, "task": "mcq", "options": options, "answer": answer,
                            "rationale": clean_text(rationale)})
        elif row.question_type == "saq":
            if not isinstance(row.answer_rationale, str) or not question:
                dropped["empty"] += 1
                continue
            records.append({**base, "task": "open", "options": [],
                            "answer": shorten(row.answer_rationale),
                            "rationale": clean_text(row.answer_rationale)})
    return pd.DataFrame(records), dropped


def build_medquad():
    data = load_dataset("lavita/MedQuAD")["train"].to_pandas()
    data = data[data.answer.notna() & data.document_source.isin(C.MEDQUAD_SOURCES)]
    data = data[data.answer.str.split().str.len() >= 8]
    return pd.DataFrame({
        # ids repeat across NIH sources, so add the source name
        "id": ("mq-" + data.document_source + "-" + data.question_id.astype(str)).tolist(),
        "source": "medquad",
        "country": "US",
        "specialty": data.question_type.fillna("Unknown").tolist(),
        "tier": "expert",
        "question": data.question.map(clean_question).tolist(),
        "task": "open",
        "options": [[] for _ in range(len(data))],
        "answer": data.answer.map(shorten).tolist(),
        "rationale": data.answer.map(clean_text).tolist(),
    })


def dedupe(frame):
    keys = frame.question.map(normalise_key)
    return frame[~keys.duplicated() & (keys.str.len() > 0)].reset_index(drop=True)


def build_ood(df, qa_train_questions, qa_test_questions):
    """Health questions (label 1) vs. general-knowledge questions (label 0)."""
    consumer = df[df.question_type == "consumer_queries"]
    health_train = [clean_question(q) for q in consumer[consumer.split == "train"].question_clean]
    health_test = [clean_question(q) for q in consumer[consumer.split == "test"].question_clean]
    health_train += list(qa_train_questions)
    health_test += list(qa_test_questions)

    webq = load_dataset("Stanford/web_questions")
    trec = load_dataset("CogComp/trec", revision="refs/convert/parquet")
    general_train = [clean_text(q) for q in webq["train"]["question"]] + \
                    [clean_text(q) for q in trec["train"]["text"]]
    general_test = [clean_text(q) for q in webq["test"]["question"]] + \
                   [clean_text(q) for q in trec["test"]["text"]]

    def frame(health, general):
        out = pd.DataFrame({"text": health + general,
                            "label": [1] * len(health) + [0] * len(general)})
        return out[~out.text.map(normalise_key).duplicated()].reset_index(drop=True)

    train, test = frame(health_train, general_train), frame(health_test, general_test)
    # drop test questions that are also in train
    test = test[~test.text.map(normalise_key).isin(set(train.text.map(normalise_key)))]
    return train, test.reset_index(drop=True)


def main():
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    df = load_afrimed()
    afri, dropped = build_afrimed_qa(df)
    n_before = len(afri)
    afri = dedupe(afri)
    dropped["duplicates"] = n_before - len(afri)

    # official split has 97% of MCQs in test, so I pool and split 80/10/10 myself
    train, rest = train_test_split(afri, test_size=0.2, stratify=afri.task, random_state=C.SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.task, random_state=C.SEED)

    medquad = dedupe(build_medquad())
    medquad = medquad[~medquad.question.map(normalise_key).isin(set(afri.question.map(normalise_key)))]
    medquad = medquad.sample(frac=1, random_state=C.SEED)
    mq_test = medquad.iloc[: C.MEDQUAD_TEST_SIZE]
    mq_train = medquad.iloc[C.MEDQUAD_TEST_SIZE: C.MEDQUAD_TEST_SIZE + C.MEDQUAD_TRAIN_SIZE]
    mq_kb_only = medquad.iloc[C.MEDQUAD_TEST_SIZE + C.MEDQUAD_TRAIN_SIZE:]

    train = pd.concat([train, mq_train]).sample(frac=1, random_state=C.SEED)
    test = pd.concat([test, mq_test])

    # knowledge base = train answers + unused MedQuAD (no val/test answers)
    kb_src = pd.concat([train, mq_kb_only])
    passages = [r.rationale if r.rationale else r.answer for r in kb_src.itertuples()]
    kb = pd.DataFrame({
        "id": kb_src.id.tolist(), "source": kb_src.source.tolist(),
        "question": kb_src.question.tolist(), "answer": kb_src.answer.tolist(),
        "passage": [shorten(t, C.CONTEXT_WORD_LIMIT) for t in passages],
    })

    for name, frame in [("train", train), ("val", val), ("test", test), ("kb", kb)]:
        frame.to_json(C.PROC_DIR / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)

    afri_train = train[train.source == "afrimedqa"]
    ood_train, ood_test = build_ood(df, afri_train.question, test.question)
    ood_train.to_json(C.PROC_DIR / "ood_train.jsonl", orient="records", lines=True, force_ascii=False)
    ood_test.to_json(C.PROC_DIR / "ood_test.jsonl", orient="records", lines=True, force_ascii=False)

    def describe(frame):
        groups = frame.groupby(["source", "task"]).size()
        return {
            "n": len(frame),
            "by_source_task": {f"{s}/{t}": int(n) for (s, t), n in groups.items()},
            "question_words_mean": round(frame.question.str.split().str.len().mean(), 1),
            "answer_words_mean": round(frame.answer.str.split().str.len().mean(), 1),
        }

    stats = {
        "afrimed_raw_rows": len(df),
        "afrimed_raw_by_type": df.question_type.value_counts().to_dict(),
        "afrimed_answered_kept": len(afri),
        "afrimed_dropped": dropped,
        "afrimed_countries": afri.country.value_counts().to_dict(),
        "afrimed_top_specialties": afri.specialty.value_counts().head(10).to_dict(),
        "mcq_options_mean": round(afri[afri.task == "mcq"].options.map(len).mean(), 2),
        "train": describe(train), "val": describe(val), "test": describe(test),
        "kb_passages": len(kb),
        "ood_train": ood_train.label.value_counts().to_dict(),
        "ood_test": ood_test.label.value_counts().to_dict(),
    }
    (C.PROC_DIR / "stats.json").write_text(json.dumps(stats, indent=2, default=int))
    print(json.dumps(stats, indent=2, default=int))


if __name__ == "__main__":
    main()

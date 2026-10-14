"""Checks for the cleaning functions and the saved splits. Run: python -m pytest tests"""
import re

from src.data_prep import clean_option, clean_question, clean_text, shorten
from src.qa_model import build_prompt, load_split


def key(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def test_clean_text_collapses_whitespace():
    assert clean_text("a \r\n\t b") == "a b"


def test_clean_question_removes_exam_numbering():
    assert clean_question("31. Which vaccine is given at birth?") == "Which vaccine is given at birth?"


def test_clean_option_removes_label_only_at_matching_position():
    assert clean_option("a. Obesity", 0) == "Obesity"
    assert clean_option("B) Obesity", 1) == "Obesity"
    # "E." at position 0 is part of the answer, not a label
    assert clean_option("E. coli", 0) == "E. coli"


def test_shorten_keeps_whole_sentences():
    text = "First sentence here. " + "Second sentence is long " * 20 + "."
    assert shorten(text, limit=10) == "First sentence here."


def test_prompt_formats():
    assert build_prompt("What is malaria?") == "Answer the medical question.\nQuestion: What is malaria?\nAnswer:"
    mcq = build_prompt("Pick one", ["x", "y"], context="some passage")
    assert mcq.startswith("Context: some passage\n")
    assert "Options: (A) x (B) y" in mcq


def test_splits_do_not_share_questions():
    train, val, test = (set(load_split(name).question.map(key)) for name in ["train", "val", "test"])
    assert not train & val and not train & test and not val & test


def test_mcq_answer_is_one_of_the_options():
    for name in ["train", "val", "test"]:
        mcq = load_split(name)
        mcq = mcq[mcq.task == "mcq"]
        assert all(answer in list(options) for answer, options in zip(mcq.answer, mcq.options))


def test_knowledge_base_has_no_test_or_validation_questions():
    import pandas as pd
    from src import config as C
    kb = set(pd.read_json(C.PROC_DIR / "kb.jsonl", lines=True).question.map(key))
    held_out = set(load_split("val").question.map(key)) | set(load_split("test").question.map(key))
    assert not kb & held_out

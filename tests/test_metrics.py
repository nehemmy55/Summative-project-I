"""Checks for the scoring functions. Run: python -m pytest tests"""
from src.metrics import bootstrap_ci, exact_match, normalise, pick_option, token_f1

OPTIONS = ["Iron deficiency", "A vitamin", "Sickle cell", "Hookworm", "E coli infection"]


def test_normalise_removes_case_punctuation_and_articles():
    assert normalise("The Malaria,  parasite!") == "malaria parasite"


def test_exact_match_ignores_formatting():
    assert exact_match("Iron deficiency.", "iron deficiency") == 1.0
    assert exact_match("iron", "iron deficiency") == 0.0


def test_token_f1():
    assert token_f1("iron deficiency", "iron deficiency") == 1.0
    assert token_f1("banana", "iron deficiency") == 0.0
    # 1 shared word: precision 1/2, recall 1/2
    assert abs(token_f1("iron overload", "iron deficiency") - 0.5) < 1e-9


def test_pick_option_exact_text():
    assert pick_option("Hookworm", OPTIONS) == 3
    assert pick_option("hookworm.", OPTIONS) == 3


def test_pick_option_letter_answers():
    assert pick_option("C", OPTIONS) == 2
    assert pick_option("(D)", OPTIONS) == 3
    assert pick_option("A. Iron", OPTIONS) == 0


def test_pick_option_text_starting_with_a_letter_is_not_a_letter_answer():
    # "A vitamin" is option B, not the letter A
    assert pick_option("A vitamin", OPTIONS) == 1
    assert pick_option("E coli infection", OPTIONS) == 4


def test_pick_option_closest_and_no_match():
    assert pick_option("Sickle cell disease", OPTIONS) == 2
    assert pick_option("banana", OPTIONS) == -1


def test_bootstrap_ci_contains_the_mean():
    low, high = bootstrap_ci([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    assert low <= 0.6 <= high

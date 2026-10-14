"""Shared paths, constants and prompt templates used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42
BASE_MODEL = "google/flan-t5-small"

AFRIMED_URL = (
    "https://raw.githubusercontent.com/intron-innovation/AfriMed-QA/master/"
    "data/afri_med_qa_15k_v2.5_phase_2_15275.csv"
)
AFRIMED_CSV = RAW_DIR / "afrimedqa_v2.5.csv"

# skip GHR and GARD (rare genetic diseases)
MEDQUAD_SOURCES = [
    "CDC", "NIDDK", "NHLBI", "MPlusHealthTopics", "NIHSeniorHealth", "NINDS", "CancerGov",
]
MEDQUAD_TRAIN_SIZE = 3000
MEDQUAD_TEST_SIZE = 200

# small limits so it can train on CPU
MAX_SOURCE_LEN = 160
MAX_SOURCE_LEN_RAG = 256
MAX_TARGET_LEN = 48
ANSWER_WORD_LIMIT = 45      # max words kept in a reference answer
CONTEXT_WORD_LIMIT = 60     # max words of the retrieved passage

OPEN_PROMPT = "Answer the medical question.\nQuestion: {question}\nAnswer:"
MCQ_PROMPT = (
    "Answer the multiple-choice medical question with the correct option.\n"
    "Question: {question}\nOptions: {options}\nAnswer:"
)
RAG_PREFIX = "Context: {context}\n"

REFUSAL = (
    "I can only help with health and medical questions. "
    "Please ask me something about symptoms, diseases, treatment or prevention."
)

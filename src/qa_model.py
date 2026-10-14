"""Prompt building, model loading and answer generation shared by training, evaluation and the app."""
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

LETTERS = "ABCDEFGH"


def load_split(name, data="afrimed+medquad"):
    df = pd.read_json(C.PROC_DIR / f"{name}.jsonl", lines=True, dtype={"id": str})
    if data == "afrimed":
        df = df[df.source == "afrimedqa"]
    return df.reset_index(drop=True)


def build_prompt(question, options=None, context=None):
    """Text given to the encoder. Options are listed as (A) ... (B) ... for MCQs."""
    if options is not None and len(options):
        listed = " ".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options))
        prompt = C.MCQ_PROMPT.format(question=question, options=listed)
    else:
        prompt = C.OPEN_PROMPT.format(question=question)
    if context:
        prompt = C.RAG_PREFIX.format(context=context) + prompt
    return prompt


def add_prompts(df, retriever=None, exclude_self=False):
    """Add a `prompt` column; with a retriever, the best passage is prepended as context."""
    df = df.copy()
    contexts = [None] * len(df)
    if retriever is not None:
        hits = retriever.search(df.question.tolist(),
                                exclude_ids=df.id.tolist() if exclude_self else None)
        contexts = [h["kb_passage"] for h in hits]
    df["prompt"] = [build_prompt(q, o, c) for q, o, c in zip(df.question, df.options, contexts)]
    return df


def load_model(path_or_name=C.BASE_MODEL):
    """Load the base model, a fully fine-tuned model, or a LoRA adapter on top of the base."""
    path = Path(path_or_name)
    run_config = {}
    if (path / "run_config.json").exists():
        run_config = json.loads((path / "run_config.json").read_text())
    elif not path.exists() and str(path_or_name) != C.BASE_MODEL:
        # model id on the Hugging Face Hub: the run settings are stored next to the weights
        from huggingface_hub import hf_hub_download
        run_config = json.loads(Path(hf_hub_download(str(path_or_name), "run_config.json")).read_text())
    if (path / "adapter_config.json").exists():
        base_name = run_config.get("base_model", C.BASE_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        # merge the LoRA weights into the base model
        model = PeftModel.from_pretrained(base, str(path)).merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(path_or_name))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(path_or_name))
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, tokenizer, run_config


@torch.no_grad()
def generate(model, tokenizer, prompts, max_source_len=C.MAX_SOURCE_LEN, batch_size=16,
             num_beams=4, progress=False):
    """Beam-search answers for a list of prompts (sorted by length to limit padding)."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs = [""] * len(prompts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tokenizer([prompts[i] for i in idx], max_length=max_source_len, truncation=True,
                        padding=True, return_tensors="pt").to(model.device)
        generated = model.generate(**enc, max_new_tokens=C.MAX_TARGET_LEN, num_beams=num_beams,
                                   no_repeat_ngram_size=3, early_stopping=True)
        for i, text in zip(idx, tokenizer.batch_decode(generated, skip_special_tokens=True)):
            outputs[i] = text.strip()
        if progress and (start // batch_size) % 10 == 0:
            print(f"  generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return outputs

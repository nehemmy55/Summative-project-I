"""Fine-tunes FLAN-T5 (full or LoRA), keeps the best checkpoint and evaluates it on the test set."""
import argparse
import json
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from . import config as C
from .evaluate import run as evaluate_run
from .qa_model import add_prompts, load_split
from .retrieval import TfidfRetriever


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_batches(df, tokenizer, batch_size, max_source_len, shuffle):
    """Tokenise into batches of similar length so little compute is wasted on padding."""
    indices = list(range(len(df)))
    if shuffle:
        random.shuffle(indices)
    # sort by length inside chunks so batches have less padding
    chunk = batch_size * 50
    batches = []
    for start in range(0, len(indices), chunk):
        part = sorted(indices[start:start + chunk], key=lambda i: len(df.prompt.iloc[i]))
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    if shuffle:
        random.shuffle(batches)
    for batch in batches:
        rows = df.iloc[batch]
        enc = tokenizer(rows.prompt.tolist(), max_length=max_source_len, truncation=True,
                        padding=True, return_tensors="pt")
        labels = tokenizer(text_target=rows.answer.tolist(), max_length=C.MAX_TARGET_LEN,
                           truncation=True, padding=True, return_tensors="pt").input_ids
        # -100 = ignored by the loss (padding)
        labels[labels == tokenizer.pad_token_id] = -100
        yield enc, labels


@torch.no_grad()
def validation_loss(model, df, tokenizer, batch_size, max_source_len):
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for enc, labels in make_batches(df, tokenizer, batch_size * 2, max_source_len, shuffle=False):
        enc, labels = enc.to(device), labels.to(device)
        n_tokens = int((labels != -100).sum())
        total += model(**enc, labels=labels).loss.item() * n_tokens
        count += n_tokens
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default=C.BASE_MODEL, help="pretrained checkpoint to adapt")
    parser.add_argument("--method", choices=["full", "lora"], default="lora")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--data", choices=["afrimed", "afrimed+medquad"], default="afrimed")
    parser.add_argument("--rag", action="store_true", help="prepend a retrieved passage as context")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=None, help="subsample, for quick tests")
    parser.add_argument("--eval-limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(C.SEED)
    # LoRA needs a higher learning rate
    lr = args.lr or (1e-3 if args.method == "lora" else 3e-4)
    max_source_len = C.MAX_SOURCE_LEN_RAG if args.rag else C.MAX_SOURCE_LEN

    retriever = TfidfRetriever() if args.rag else None
    train = load_split("train", args.data)
    if args.max_train:
        train = train.sample(args.max_train, random_state=C.SEED).reset_index(drop=True)
    # a train question must not retrieve its own answer
    train = add_prompts(train, retriever, exclude_self=True)
    val = add_prompts(load_split("val"), retriever)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if args.method == "lora":
        # LoRA on the attention query and value layers
        lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                          target_modules=["q", "v"], task_type="SEQ_2_SEQ_LM")
        model = get_peft_model(model, lora)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {len(train)} train examples, trainable params {trainable:,} "
          f"({100 * trainable / total:.2f}% of {total:,}), lr {lr}, device {device}", flush=True)

    steps_per_epoch = int(np.ceil(len(train) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    out_dir = C.MODEL_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_val, step, start = [], float("inf"), 0, time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = []
        for enc, labels in make_batches(train, tokenizer, args.batch_size, max_source_len, shuffle=True):
            loss = model(**enc.to(device), labels=labels.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running.append(loss.item())
            step += 1
            if step % 50 == 0:
                elapsed = time.time() - start
                eta = elapsed / step * (total_steps - step) / 60
                print(f"  step {step}/{total_steps} loss {np.mean(running[-50:]):.3f} "
                      f"({elapsed / step:.2f}s/step, ~{eta:.0f} min left)", flush=True)
        val_loss = validation_loss(model, val, tokenizer, args.batch_size, max_source_len)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(running)), 4),
                        "val_loss": round(val_loss, 4)})
        print(f"epoch {epoch}: train loss {history[-1]['train_loss']} val loss {val_loss:.4f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out_dir)   # LoRA saves only the adapter
            tokenizer.save_pretrained(out_dir)

    run_config = {**vars(args), "lr": lr, "base_model": args.model, "device": device, "trainable_params": trainable,
                  "total_params": total, "train_examples": len(train), "history": history,
                  "best_val_loss": round(best_val, 4),
                  "train_minutes": round((time.time() - start) / 60, 1)}
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    del model

    metrics = evaluate_run(str(out_dir), args.name, limit=args.eval_limit)
    (C.RESULTS_DIR / args.name / "run_config.json").write_text(json.dumps(run_config, indent=2))
    return metrics


if __name__ == "__main__":
    main()

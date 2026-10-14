"""Full pipeline for the app: domain gate, retriever, then the QA model."""
import os
import sys

from . import config as C
from .domain_gate import is_health_question, load_gate
from .metrics import pick_option
from .qa_model import build_prompt, generate, load_model
from .retrieval import TfidfRetriever

# local folder first, otherwise the Hugging Face repo
LOCAL_MODEL = C.MODEL_DIR / "final"
HUB_MODEL = os.environ.get("QA_MODEL_ID", "ishimwe5555/afrimed-qa")


class MedicalAssistant:
    def __init__(self, model_path=None):
        path = model_path or (LOCAL_MODEL if LOCAL_MODEL.exists() else HUB_MODEL)
        self.model, self.tokenizer, self.run_config = load_model(path)
        self.use_rag = bool(self.run_config.get("rag"))
        self.gate = load_gate()
        self.retriever = TfidfRetriever()

    def answer(self, question, options=None):
        """Return the answer together with everything the app shows about how it was produced."""
        question = " ".join(question.split())
        options = [o.strip() for o in (options or []) if o.strip()]
        result = {"question": question, "in_domain": False, "p_health": 0.0, "answer": C.REFUSAL,
                  "picked_option": None, "reference": None}
        if len(question.split()) < 3:
            # the gate is not reliable on very short inputs like "hello"
            result["answer"] = "Please type a complete health question, for example: " \
                               "'What are the symptoms of malaria?'"
            return result

        in_domain, probability = is_health_question(self.gate, question)
        result["p_health"] = probability
        # if options are given, treat it as an exam question
        if not in_domain and not options:
            return result
        result["in_domain"] = True

        hit = self.retriever.search([question])[0]
        result["reference"] = hit
        context = hit["kb_passage"] if self.use_rag else None
        prompt = build_prompt(question, options, context)
        max_len = C.MAX_SOURCE_LEN_RAG if self.use_rag else C.MAX_SOURCE_LEN
        text = generate(self.model, self.tokenizer, [prompt], max_len)[0]
        result["answer"] = text
        if options:
            idx = pick_option(text, options)
            result["picked_option"] = options[idx] if idx >= 0 else None
        return result


if __name__ == "__main__":
    assistant = MedicalAssistant()
    for query in sys.argv[1:]:
        out = assistant.answer(query)
        print(f"Q: {query}\nA: {out['answer']}\n(p_health={out['p_health']:.2f})\n")

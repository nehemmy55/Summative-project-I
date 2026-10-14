"""TF-IDF retriever over the knowledge base."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config as C


class TfidfRetriever:
    def __init__(self, kb=None):
        self.kb = kb if kb is not None else pd.read_json(C.PROC_DIR / "kb.jsonl", lines=True, dtype={"id": str})
        self.kb = self.kb.reset_index(drop=True)
        # bigrams keep names like "sickle cell" together
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                          stop_words="english", min_df=1)
        text = self.kb.question + " " + self.kb.passage
        self.matrix = self.vectorizer.fit_transform(text)

    def search(self, queries, exclude_ids=None, batch_size=512):
        """Best KB entry and its cosine similarity for each query (exclude_ids hides a question's own entry)."""
        ids = self.kb.id.to_numpy()
        results = []
        for start in range(0, len(queries), batch_size):
            batch = queries[start:start + batch_size]
            # rows are L2-normalised, so dot product = cosine similarity
            sims = (self.vectorizer.transform(batch) @ self.matrix.T).toarray()
            for offset, row in enumerate(sims):
                if exclude_ids is not None:
                    row[ids == exclude_ids[start + offset]] = -1.0
                best = int(row.argmax())
                hit = self.kb.iloc[best]
                results.append({"kb_id": hit.id, "kb_question": hit.question, "kb_answer": hit.answer,
                                "kb_passage": hit.passage, "score": float(row[best])})
        return results

# src/data/retriever.py
import os
import json
import faiss

from src.llm.client import embed_text

DOCS_NAME = "cbt_docs.jsonl"
INDEX_NAME = "cbt.index"


class CBTRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.docs_path = os.path.join(cfg.INDEX_DIR, DOCS_NAME)
        self.index_path = os.path.join(cfg.INDEX_DIR, INDEX_NAME)

        if not os.path.exists(self.docs_path) or not os.path.exists(self.index_path):
            raise FileNotFoundError(
                "Index belum dibuat. Jalankan ensure_index() atau build_index() dulu."
            )

        self.index = faiss.read_index(self.index_path)

        self.docs = []
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.docs.append(json.loads(line))

        if len(self.docs) == 0:
            raise RuntimeError(
                f"Docs kosong ({self.docs_path}). Pastikan ingest berhasil membangun docs."
            )

    def _ensure_text(self, d: dict) -> str:
        """
        Pastikan selalu ada field 'text' untuk prompt.
        """
        text = (d.get("text") or "").strip()
        if text:
            # normalisasi label legacy kalau ada
            return text.replace("Patient:", "Client:")

        q = (d.get("query") or d.get("patient") or "").strip()
        r = (d.get("response") or d.get("therapist") or "").strip()
        if q or r:
            return f"Client: {q}\nTherapist: {r}".strip()

        return ""

    def search(self, query: str, k: int = 5, dataset_filter: str | None = None):
        """
        dataset_filter: "HOPE" / "HQC" / None (gabungan)
        Return list[dict] yang sudah siap untuk build_messages().
        """
        query = (query or "").strip()
        if not query:
            return []

        # embed query
        qvec = embed_text(query, model=self.cfg.EMBED_MODEL)  # (1, dim), normalized

        # adaptif: jangan minta probe_k melebihi total vector di index
        ntotal = int(getattr(self.index, "ntotal", 0))
        if ntotal <= 0:
            return []

        # kalau pakai filter, tarik kandidat lebih banyak dulu
        # tapi jangan melebihi ntotal
        base_probe = max(k * 5, 25) if dataset_filter else k
        probe_k = min(base_probe, ntotal)

        scores, idxs = self.index.search(qvec, probe_k)

        out = []
        seen = set()  # untuk skip duplikat (session_id+query) atau text
        for score, idx in zip(scores[0], idxs[0]):
            # FAISS bisa mengembalikan -1
            if idx is None or int(idx) < 0:
                continue

            idx = int(idx)
            if idx >= len(self.docs):
                continue

            d = self.docs[idx]

            if dataset_filter and d.get("dataset") != dataset_filter:
                continue

            text = self._ensure_text(d)
            if not text:
                continue

            # dedup key
            key = (d.get("dataset"), d.get("session_id"), d.get("query"), d.get("response"))
            if key in seen:
                continue
            seen.add(key)

            out.append({
                "score": float(score),
                "dataset": d.get("dataset"),
                "session_id": d.get("session_id"),
                "source_file": d.get("source_file"),

                "query": d.get("query"),
                "response": d.get("response"),
                "text": text,
            })

            if len(out) >= k:
                break

        return out

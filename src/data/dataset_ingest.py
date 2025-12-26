# src/data/dataset_ingest.py
import os
import glob
import json
import re
import numpy as np
import pandas as pd
import faiss

from src.llm.client import embed_texts

DOCS_NAME = "cbt_docs.jsonl"
INDEX_NAME = "cbt.index"


def _clean(s: str) -> str:
    s = str(s).replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _make_pair_docs(rows, session_id: str, source_file: str, dataset_name: str):
    """
    rows: list of {"role": "C"|"T", "utterance": "..."}
    Build Client->Therapist pairs.
    """
    docs = []
    for i in range(len(rows) - 1):
        cur = rows[i]
        nxt = rows[i + 1]

        if cur["role"] == "C" and nxt["role"] == "T":
            client = _clean(cur["utterance"])
            therapist = _clean(nxt["utterance"])
            if not client or not therapist:
                continue

            docs.append({
                "dataset": dataset_name,
                "session_id": session_id,
                "source_file": source_file,

                # ✅ yang di-embed (untuk retrieval)
                "query": client,

                # ✅ target contoh respons
                "response": therapist,

                # ✅ context untuk prompt
                "text": f"Client: {client}\nTherapist: {therapist}",
            })
    return docs


# -------------------------
# HOPE loader (CSV)
# -------------------------
def _load_hope_csv_pairs(hope_dir: str):
    """
    HOPE CSV expected columns: ID, Type, Utterance
    Type: P (patient) / T (therapist)
    Kita map: P -> C (Client), T -> T (Therapist)
    """
    docs = []
    csv_files = sorted(glob.glob(os.path.join(hope_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"Tidak ada file .csv di folder HOPE: {hope_dir}")

    for fp in csv_files:
        df = pd.read_csv(fp)

        required = {"ID", "Type", "Utterance"}
        if not required.issubset(set(df.columns)):
            raise ValueError(
                f"Kolom CSV HOPE tidak sesuai di {os.path.basename(fp)}. "
                f"Harus ada {required}, ketemu {set(df.columns)}"
            )

        rows = []
        for r in df.to_dict(orient="records"):
            role = _clean(r["Type"]).upper()
            utt = _clean(r["Utterance"])
            if not utt:
                continue

            if role == "P":
                role = "C"
            elif role == "T":
                role = "T"
            else:
                continue

            rows.append({"role": role, "utterance": utt})

        # contoh: ID = "97_0" => session_id "97"
        if len(df) > 0:
            sid = _clean(df.iloc[0]["ID"]).split("_")[0]
        else:
            sid = os.path.splitext(os.path.basename(fp))[0]

        docs.extend(_make_pair_docs(
            rows=rows,
            session_id=sid,
            source_file=os.path.basename(fp),
            dataset_name="HOPE"
        ))

    return docs


# -------------------------
# HQC loader (plain text: T: ..., C: ...)
# -------------------------
def _load_hqc_pairs(hqc_dir: str):
    """
    Format sesuai contoh:
    T:\tHello ...
    C:\tHi ...
    """
    docs = []

    # HQC file bisa tanpa ekstensi → ambil semua file (bukan folder)
    all_files = sorted([p for p in glob.glob(os.path.join(hqc_dir, "*")) if os.path.isfile(p)])
    if not all_files:
        raise FileNotFoundError(f"Tidak ada file di folder HQC: {hqc_dir}")

    # cocokkan "T: ..." atau "C: ..." dengan kemungkinan tab/spasi
    pat = re.compile(r"^\s*([TC])\s*:\s*(.*)$")

    for fp in all_files:
        sid = os.path.splitext(os.path.basename(fp))[0]

        rows = []
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for raw in f.read().splitlines():
                raw = raw.strip()
                if not raw:
                    continue

                raw = raw.replace("\t", " ")
                m = pat.match(raw)
                if not m:
                    continue

                role = m.group(1)  # "T" or "C"
                utt = _clean(m.group(2))
                if utt:
                    rows.append({"role": role, "utterance": utt})

        docs.extend(_make_pair_docs(
            rows=rows,
            session_id=sid,
            source_file=os.path.basename(fp),
            dataset_name="HQC"
        ))

    return docs


def _collect_all_docs(cfg):
    docs = []

    # HOPE wajib (kalau foldernya ada)
    if not os.path.isdir(cfg.HOPE_DIR):
        raise FileNotFoundError(f"HOPE_DIR tidak ditemukan: {cfg.HOPE_DIR}")
    docs.extend(_load_hope_csv_pairs(cfg.HOPE_DIR))

    # HQC opsional
    if hasattr(cfg, "HQC_DIR") and os.path.isdir(cfg.HQC_DIR):
        docs.extend(_load_hqc_pairs(cfg.HQC_DIR))
    else:
        print("⚠️ HQC_DIR tidak ditemukan / tidak dipakai. Index hanya HOPE.")

    return docs


def build_index(cfg):
    os.makedirs(cfg.INDEX_DIR, exist_ok=True)

    docs_path = os.path.join(cfg.INDEX_DIR, DOCS_NAME)
    index_path = os.path.join(cfg.INDEX_DIR, INDEX_NAME)

    docs = _collect_all_docs(cfg)
    if len(docs) == 0:
        raise RuntimeError("Tidak ada pasangan C->T yang terbentuk dari dataset.")

    # ✅ Embedding dari client query saja
    queries = [d["query"] for d in docs]

    all_vecs = []
    BATCH = 128
    for start in range(0, len(queries), BATCH):
        batch = queries[start:start + BATCH]
        vecs = embed_texts(batch, model=cfg.EMBED_MODEL)  # normalized for cosine
        all_vecs.append(vecs)
        print(f"Embedded {min(start+BATCH, len(queries))}/{len(queries)}")

    vectors = np.vstack(all_vecs).astype("float32")
    dim = vectors.shape[1]

    # cosine similarity (normalize + inner product)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # save docs
    with open(docs_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # save index
    faiss.write_index(index, index_path)

    print(f"✅ Saved docs:  {docs_path}")
    print(f"✅ Saved index: {index_path}")
    print(f"✅ Total pairs: {len(docs)}")


def ensure_index(cfg, force_rebuild: bool = False):
    docs_path = os.path.join(cfg.INDEX_DIR, DOCS_NAME)
    index_path = os.path.join(cfg.INDEX_DIR, INDEX_NAME)

    if (not force_rebuild) and os.path.exists(docs_path) and os.path.exists(index_path):
        return

    print("ℹ️ Building index dari dataset HOPE + HQC...")
    build_index(cfg)

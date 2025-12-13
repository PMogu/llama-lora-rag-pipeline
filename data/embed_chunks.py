import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = BASE_DIR / "data" / "raw" / "chunk" / "neuroimaging.jsonl"

RAG_DIR = BASE_DIR / "data" / "RAG"
RAG_DIR.mkdir(exist_ok=True)

INDEX_PATH = RAG_DIR / "index.faiss"
META_PATH  = RAG_DIR / "meta.json"

MODEL_PATH = BASE_DIR / "models" / "embedding" / "bge-m3"

print("[INFO] Loading embedding model...")
model = SentenceTransformer(str(MODEL_PATH))

texts = []
metadata = []

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        record = json.loads(line)

        assert "text" in record, f"Missing text field at line {i}"
        text = record["text"]
        assert isinstance(text, str), f"text is not str at line {i}"
        text = text.strip()
        assert text != "", f"Empty text at line {i}"

        texts.append(text)

        metadata.append({
            "id": record.get("id", f"chunk_{i}"),
            "source": record.get("source", "unknown"),
            "chunk_index": record.get("chunk_index", i),
            "text": text
        })

print(f"[INFO] Loaded {len(texts)} chunks")

# Embedding
print("[INFO] Computing embeddings...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

assert embeddings.shape[0] == len(metadata), "Embedding / metadata size mismatch"

print(f"[INFO] Embedding shape: {embeddings.shape}")

# FAISS
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print(f"[INFO] FAISS index size: {index.ntotal}")

faiss.write_index(index, str(INDEX_PATH))

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("[DONE] RAG embedding build complete")
print(f"  → Index saved to {INDEX_PATH}")
print(f"  → Metadata saved to {META_PATH}")

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


MODEL_NAME = str(MODEL_PATH)

# Load embedding model
print("[INFO] Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# Load chunks
texts = []
metadata = []

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        texts.append(record["text"])
        metadata.append({
            "id": record["id"],
            "source": record["source"],
            "chunk_index": record["chunk_index"]
        })

print(f"[INFO] Loaded {len(texts)} chunks")

# Compute embeddings
print("[INFO] Computing embeddings...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"[INFO] Embedding shape: {embeddings.shape}")

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)   # cosine similarity

index.add(embeddings)

print(f"[INFO] FAISS index size: {index.ntotal}")

# Save index & metadata
faiss.write_index(index, str(INDEX_PATH))

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("[DONE] RAG embedding build complete")
print(f"  → Index saved to {INDEX_PATH}")
print(f"  → Metadata saved to {META_PATH}")

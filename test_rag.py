import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

BASE_MODEL_PATH = "/Users/Chen/Desktop/LLM/models/Llama-3.2-3B-Instruct"
EMBED_MODEL_PATH = "/Users/Chen/Desktop/LLM/models/embedding/bge-m3"

BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH = BASE_DIR / "models" / "Llama-3.2-3B-Instruct"
ADAPTER_PATH = BASE_DIR / "models" / "embedding" / "bge-m3"

RAG_DIR = Path("/Users/Chen/Desktop/LLM/data/RAG")
INDEX_PATH = RAG_DIR / "index.faiss"
META_PATH  = RAG_DIR / "meta.json"

TOP_K = 5

print("[INFO] Loading base model...")
model, tokenizer = load(BASE_MODEL_PATH) # load llm

print("[INFO] Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_PATH) # load embedding model

print("[INFO] Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH)) # load FAISS index & metadata

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"[INFO] Loaded {index.ntotal} vectors")

def apply_instruct_template(prompt: str) -> str:
    return f"""<|begin_of_text|>
<|user|>
{prompt}
<|assistant|>
"""


def build_rag_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)
    )

    return f"""You are a helpful, accurate biology assistant.
Use the following reference material to answer the question.
If the answer is not contained in the material, say you are not sure.

{context_block}

Question:
{question}
"""

def retrieve_context(query: str, k: int = TOP_K) -> list[str]:
    query_emb = embed_model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(query_emb, k)

    contexts = []
    for idx in indices[0]:
        if idx < 0:
            continue

        meta = metadata[idx]
        source = meta["source"]
        chunk_id = meta["id"]

        contexts.append(
            f"(Source: {source}, Chunk: {chunk_id})"
        )

    return contexts

test_questions = [
    "What are the respective roles of the slice-selection gradient, phase-encoding gradient, and frequency-encoding gradient in MRI image formation?",
    "Why does diffusion tensor imaging (DTI) show anisotropic diffusion in white matter, and how is this property used to infer fiber tract orientation?",
    "How do repetition time (TR) and echo time (TE) differ between T1-weighted and T2-weighted MRI, and why do these choices affect tissue contrast?",
    "What are the respective roles of the slice-selection gradient, phase-encoding gradient, and frequency-encoding gradient in MRI image formation?",
    "What are the main trade-offs between single-voxel magnetic resonance spectroscopy (MRS) and magnetic resonance spectroscopy imaging (MRSI)?",
]


gen_kwargs = dict(
    max_tokens=256
)

for q in test_questions:
    print("=" * 100)
    print("QUESTION:\n", q)

    contexts = retrieve_context(q)

    rag_prompt = build_rag_prompt(q, contexts)

    output = generate(
        model,
        tokenizer,
        prompt=apply_instruct_template(rag_prompt),
        **gen_kwargs
    )

    print("\n--- RAG OUTPUT ---\n")
    print(output)

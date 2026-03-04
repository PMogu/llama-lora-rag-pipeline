import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate

BASE_DIR = Path(__file__).resolve().parent.parent

BASE_MODEL_PATH  = BASE_DIR / "models" / "Llama-3.2-3B-Instruct"
ADAPTER_PATH     = BASE_DIR / "adapters" / "BIOL10010"
EMBED_MODEL_PATH = BASE_DIR / "models" / "embedding" / "bge-m3"
RAG_DIR          = BASE_DIR / "data" / "RAG"

# Strings for 3rd-party libs
BASE_MODEL_NAME  = str(BASE_MODEL_PATH)
ADAPTER_NAME     = str(ADAPTER_PATH)
EMBED_MODEL_NAME = str(EMBED_MODEL_PATH)
RAG_DIR_STR      = str(RAG_DIR)

INDEX_PATH = RAG_DIR / "index.faiss"
META_PATH  = RAG_DIR / "meta.json"

TOP_K = 5 # 每次检索取最相似的5条
MAX_TOKENS = 256

# RAG & LoRA (can be turned on/off)
USE_RAG = True
USE_LORA = False

chat_mode = False
if chat_mode:
    USE_RAG = False
    USE_LORA = False

print("[INFO] Loading LLM...")

if USE_LORA:
    model, tokenizer = load(
        BASE_MODEL_PATH,
        adapter_path=ADAPTER_PATH
    )
else:
    model, tokenizer = load(BASE_MODEL_PATH)

print("[INFO] LLM loaded")

if USE_RAG:
    print("[INFO] Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[INFO] Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[INFO] RAG index loaded ({index.ntotal} chunks)")
    
def apply_instruct_template(prompt: str) -> str:
    return f"Assistant:\n{prompt}\n\n"

def build_chat_prompt(question: str) -> str:
    return f"""You are a helpful, accurate research assistant, can be also used for daily chatting.

Question:
{question}
"""

def build_rag_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)
    )

    return f"""You are a helpful, accurate research assistant.
Use the reference material below to answer the question.
If the answer cannot be found in the material, say you are not sure.

{context_block}

Question:
{question}
"""

# RAG
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

        chunk_text = meta.get("text", "")
        source = meta.get("source", "unknown")
        chunk_id = meta.get("id", idx)

        contexts.append(
            f"{chunk_text}\n(Source: {source}, Chunk: {chunk_id})"
        )

    return contexts

print("\n=== Interactive Chat (type 'exit' to quit) ===\n")

while True:
    user_input = input("User:\n").strip()
    print("\n")

    if user_input.lower() in {"exit", "quit", "q", "bye"}:
        print("Bye 👋\n")
        break

    # ---- RAG ----
    if USE_RAG:
        contexts = retrieve_context(user_input)
        final_prompt = build_rag_prompt(user_input, contexts)
    else:
        final_prompt = build_chat_prompt(user_input)

    # ---- Use instruct template ----
    prompt_to_send = apply_instruct_template(final_prompt)  # Instruct format

    # ---- Generate ----
    output = generate(
        model,
        tokenizer,
        prompt=prompt_to_send,
        max_tokens=MAX_TOKENS
    )


    #print("\nAssistant:\n")
    print(output)
    print("\n" + "-" * 80 + "\n")
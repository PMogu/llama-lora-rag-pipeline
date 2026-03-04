# research_assistant_v2.py
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

# Strings for 3rd-party libs (kept for compatibility / debugging)
BASE_MODEL_NAME  = str(BASE_MODEL_PATH)
ADAPTER_NAME     = str(ADAPTER_PATH)
EMBED_MODEL_NAME = str(EMBED_MODEL_PATH)
RAG_DIR_STR      = str(RAG_DIR)

INDEX_PATH = RAG_DIR / "index.faiss"
META_PATH  = RAG_DIR / "meta.json"

TOP_K = 5      # 每次检索取最相似的5条
MAX_TOKENS = 256

# RAG & LoRA (can be turned on/off)
USE_RAG = False
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

# -----------------------------
# RAG init (optional)
# -----------------------------
if USE_RAG:
    print("[INFO] Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[INFO] Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[INFO] RAG index loaded ({index.ntotal} chunks)")


# -----------------------------
# Official chat-template aligned prompting
# -----------------------------
def build_messages_chat(question: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a helpful, accurate research assistant, can be also used for daily chatting."
        },
        {"role": "user", "content": question},
    ]


def build_messages_rag(question: str, contexts: list[str]) -> list[dict]:
    context_block = "\n\n".join(
        f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)
    )

    user_content = (
        "Use the reference material below to answer the question.\n"
        "If the answer cannot be found in the material, say you are not sure.\n\n"
        f"{context_block}\n\n"
        f"Question:\n{question}"
    )

    return [
        {"role": "system", "content": "You are a helpful, accurate research assistant."},
        {"role": "user", "content": user_content},
    ]


def render_prompt_from_messages(messages: list[dict]) -> str:
    """
    Align with the model's official tokenizer chat_template when available.
    For Llama 3.x Instruct, this produces the <|start_header_id|>... format and
    appends the assistant generation header when add_generation_prompt=True.
    """
    # Prefer official chat_template
    has_apply = hasattr(tokenizer, "apply_chat_template")
    has_template = bool(getattr(tokenizer, "chat_template", None))

    if has_apply and has_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,              # return prompt string
            add_generation_prompt=True,  # append assistant header
        )

    # Fallback (should rarely happen)
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}:\n{m['content']}".strip())
    return "\n\n".join(parts) + "\n\nASSISTANT:\n"


# -----------------------------
# RAG retrieval
# -----------------------------
def retrieve_context(query: str, k: int = TOP_K) -> list[str]:
    query_emb = embed_model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_emb, k)

    contexts = []
    for idx in indices[0]:
        if idx < 0:
            continue

        meta = metadata[idx]
        chunk_text = meta.get("text", "")
        source = meta.get("source", "unknown")
        chunk_id = meta.get("id", idx)

        contexts.append(f"{chunk_text}\n(Source: {source}, Chunk: {chunk_id})")

    return contexts


print("\n=== Interactive Chat (type 'exit' to quit) ===\n")

while True:
    user_input = input("User:\n").strip()
    #print("\n")

    if user_input.lower() in {"exit", "quit", "q", "bye"}:
        print("Bye 👋\n")
        break

    # ---- Build messages (RAG or normal chat) ----
    if USE_RAG:
        contexts = retrieve_context(user_input)
        messages = build_messages_rag(user_input, contexts)
    else:
        messages = build_messages_chat(user_input)

    # ---- Render prompt via official chat_template ----
    prompt_to_send = render_prompt_from_messages(messages)

    # ---- Generate ----
    output = generate(
        model,
        tokenizer,
        prompt=prompt_to_send,
        max_tokens=MAX_TOKENS
    )

    print("\nAssistant:")
    print(output.strip())
    print("\n" + "-" * 80 + "\n")
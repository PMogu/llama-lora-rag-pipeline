import json
import re
from pathlib import Path
import tiktoken

BASE_DIR = Path(__file__).resolve().parent.parent

TXT_DIR = BASE_DIR / "data" / "raw" / "pdf2txt"
OUT_DIR  = BASE_DIR / "data" / "raw" / "chunk"

OUT_DIR.mkdir(exist_ok=True)

OUT_FILE = OUT_DIR / "neuroimaging.jsonl"

CHUNK_SIZE = 400
OVERLAP = 80

enc = tiktoken.get_encoding("cl100k_base")

CONTENT_PATTERN = re.compile(
    r"<content>(.*?)</content>",
    re.DOTALL | re.IGNORECASE
)

# Token-based chunking
def token_chunk(text: str):
    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append(chunk_text)
        start += CHUNK_SIZE - OVERLAP

    return chunks

# Hybrid chunking:
# <content> blocks > token chunks
def chunk_text(text: str):
    chunks = []
    idx = 0
    last_end = 0

    # Iterate over all <content>...</content>
    for match in CONTENT_PATTERN.finditer(text):
        start, end = match.span()

        # ---- 1. normal text before <content> ----
        normal_text = text[last_end:start].strip()
        if normal_text:
            for chunk in token_chunk(normal_text):
                chunks.append((idx, chunk))
                idx += 1

        # ---- 2. <content> block itself (atomic) ----
        content_text = match.group(1).strip()
        if content_text:
            chunks.append((idx, content_text))
            idx += 1

        last_end = end

    # ---- 3. remaining tail text ----
    tail_text = text[last_end:].strip()
    if tail_text:
        for chunk in token_chunk(tail_text):
            chunks.append((idx, chunk))
            idx += 1

    return chunks

def main():
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for txt_file in TXT_DIR.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8")
            chunks = chunk_text(text)

            for idx, chunk in chunks:
                record = {
                    "id": f"{txt_file.stem}_{idx:04d}",
                    "text": chunk,
                    "source": txt_file.name,
                    "chunk_index": idx
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[OK] {txt_file.name} → {len(chunks)} chunks")

if __name__ == "__main__":
    main()
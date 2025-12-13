import json
from pathlib import Path
import tiktoken

BASE_DIR = Path(__file__).resolve().parent.parent

TXT_DIR = BASE_DIR / "data" / "raw" / "pdf2txt" / "neuroimaging"
OUT_DIR  = BASE_DIR / "data" / "raw" / "chunk"

OUT_DIR.mkdir(exist_ok=True)

OUT_FILE = OUT_DIR / "neuroimaging.jsonl"

CHUNK_SIZE = 400
OVERLAP = 80

enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str):
    tokens = enc.encode(text)
    chunks = []

    start = 0
    idx = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append((idx, chunk_text))
        idx += 1
        start += CHUNK_SIZE - OVERLAP

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
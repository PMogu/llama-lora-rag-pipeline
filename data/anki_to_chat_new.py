import json
import random
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_PATH = BASE_DIR / "raw" / "anki1.txt"
OUT_DIR  = BASE_DIR / "train3"

TRAIN_RATIO = 0.95
RANDOM_SEED = 42

SYSTEM_MESSAGE = "You are a helpful, accurate, and concise biology assistant."

OUT_DIR.mkdir(exist_ok=True)

# SYSTEM extraction
SYSTEM_PATTERN = re.compile(
    r"<SYSTEM>(.*?)</SYSTEM>",
    re.DOTALL | re.IGNORECASE
)

def extract_system_and_clean_answer(text, default_system):
    """
    从 answer 中提取 <SYSTEM>...</SYSTEM>
    若不存在，则使用 default_system
    返回 (system_message, cleaned_answer)
    """
    match = SYSTEM_PATTERN.search(text)

    if match and match.group(1).strip():
        system_message = match.group(1).strip()
        cleaned_text = SYSTEM_PATTERN.sub("", text).strip()
    else:
        system_message = default_system
        cleaned_text = text.strip()

    return system_message, cleaned_text

# Load Anki txt
samples = []

with open(RAW_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # 跳过空行 & metadata
        if not line or line.startswith("#"):
            continue

        # Anki 默认是 tab 分隔
        if "\t" not in line:
            continue

        question, answer = line.split("\t", 1)

        question = question.strip()
        answer   = answer.strip()

        if not question or not answer:
            continue

        system_msg, clean_answer = extract_system_and_clean_answer(
            answer,
            SYSTEM_MESSAGE
        )

        samples.append({
            "messages": [
                {
                    "role": "system",
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": clean_answer
                }
            ]
        })

print(f"Loaded {len(samples)} valid Anki cards")

# Shuffle
random.seed(RANDOM_SEED)
random.shuffle(samples)

# Train / valid split
split_idx = int(len(samples) * TRAIN_RATIO)

train_data = samples[:split_idx]
valid_data = samples[split_idx:]

print(f"Train samples: {len(train_data)}")
print(f"Valid samples: {len(valid_data)}")

# Write jsonl
def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(OUT_DIR / "train.jsonl", train_data)
write_jsonl(OUT_DIR / "valid.jsonl", valid_data)

print("Done. Anki chat-style JSONL written to train3/")

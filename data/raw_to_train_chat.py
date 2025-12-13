import json
import random
from pathlib import Path
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "medical_flashcards.json"
OUT_DIR  = BASE_DIR / "data" / "LoRA_train" / "train2"
TRAIN_RATIO = 0.95
RANDOM_SEED = 42

SYSTEM_MESSAGE = "You are a helpful, accurate, and concise medical assistant."

OUT_DIR.mkdir(exist_ok=True)

# Load raw data
with open(RAW_PATH, "r", encoding="utf-8") as file:
    data = json.load(file)  # list[dict]

print(f"Loaded {len(data)} raw samples")

# Process samples
processed = []

for sample in data:
    user_content = sample.get("input", "").strip()
    assistant_content = sample.get("output", "").strip()

    # filter empty values
    if not user_content or not assistant_content:
        continue

    processed.append({
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    })

print(f"After filtering: {len(processed)} samples")

# Shuffle
random.seed(RANDOM_SEED)
random.shuffle(processed)

# Train / valid split
split_idx = int(len(processed) * TRAIN_RATIO)

train_data = processed[:split_idx]
valid_data = processed[split_idx:]

print(f"Train samples: {len(train_data)}")
print(f"Valid samples: {len(valid_data)}")

# Write jsonl
def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

write_jsonl(OUT_DIR / "train.jsonl", train_data)
write_jsonl(OUT_DIR / "valid.jsonl", valid_data)

print("Done. Chat-style JSONL files written to train2/")

import json
import random
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "medical_flashcards.json"
OUT_DIR  = BASE_DIR / "data" / "LoRA_train" / "train"

TRAIN_RATIO = 0.95
RANDOM_SEED = 42

OUT_DIR.mkdir(exist_ok=True)

with open(RAW_PATH, "r") as file:
    data = json.load(file) # data is a Python list of dictionary

print(f"Loaded {len(data)} raw samples")

processed = []

for sample in data:
    prompt     = sample["input"]
    completion = sample["output"]

    if not prompt or not completion:
        continue

    processed.append({"prompt": prompt, "completion": completion})

print(f"After filtering: {len(processed)} samples")


# Shuffle
random.seed(RANDOM_SEED)
random.shuffle(processed)

split_idx = int(len(processed) * TRAIN_RATIO)

train_data = processed[:split_idx]
valid_data = processed[split_idx:]

print(f"Train samples: {len(train_data)}")
print(f"Valid samples: {len(valid_data)}")

def write_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

write_jsonl(OUT_DIR / "train.jsonl", train_data)
write_jsonl(OUT_DIR / "valid.jsonl", valid_data)

print("Done. JSONL files written to train/")
from mlx_lm import load, generate
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH = BASE_DIR / "models" / "Llama-3.2-3B-Instruct"
ADAPTER_PATH = BASE_DIR / "adapters" / "BIOL10010"

base_model, tokenizer = load(BASE_MODEL_PATH) # load base model

lora_model, _ = load(
    BASE_MODEL_PATH,
    adapter_path=ADAPTER_PATH,
) # load base + LoRA

test_prompts = [
    "What is Q10?",
    "List the “Evil Quartet” of extinction drivers.",
    "What traits allow a pathogen to drive extinction?",
    "What are r- and K-selected species?",
    "What features do carnivores have?",
    "Define Species.",
]

def apply_instruct_template(prompt: str) -> str:
    return f"""<|begin_of_text|>
<|user|>
{prompt}
<|assistant|>
"""

gen_kwargs = dict(
    max_tokens=256
)

for p in test_prompts:
    print("=" * 100)
    print("PROMPT:\n", p)

    base_out = generate(
        base_model,
        tokenizer,
        prompt=apply_instruct_template(p),
        **gen_kwargs
    )

    print("\n--- BASE MODEL OUTPUT ---\n")
    print(base_out)

    lora_out = generate(
        lora_model,
        tokenizer,
        prompt=apply_instruct_template(p),
        **gen_kwargs
    )

    print("\n--- BASE + LoRA OUTPUT ---\n")
    print(lora_out)

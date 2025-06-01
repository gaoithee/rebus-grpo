from unsloth import FastLanguageModel
from reward_funcs import parse_generation

max_seq_length = 1248 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

model_type = "qwen-2.5" # llama, phi-3, gemma

model, tokenizer = FastLanguageModel.from_pretrained(model_name = "saracandu/qwen3-4b-rebus-solver-adapters")
FastLanguageModel.for_inference(model)

from datasets import load_dataset

eval_dataset = load_dataset('gsarti/eureka-rebus', 'llm_sft', data_files=["id_test.jsonl", "ood_test.jsonl"], split = "train")

import re
import pandas as pd
from tqdm import tqdm

stop_token_id = model.config.eos_token_id
if model_type == "gemma":
    stop_token = "<|eot_id|>"
    stop_token_id = tokenizer.encode(stop_token)[0]

if model_type == "llama":
    tokenizer.padding_side = "right"
elif model_type in ("phi-3", "gemma", "qwen-2.5"):
    tokenizer.padding_side = "left"

results = []

for ex_idx in tqdm(range(0, len(eval_dataset)), desc="Processing examples"):
    example = eval_dataset[ex_idx]["conversations"][0]
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example["value"]}
        ],
        add_generation_prompt=True,
        return_tensors = "pt",
        padding=True,
        truncation=True,
    )
    inputs = inputs.to('cuda:0')
    outputs = model.generate(input_ids = inputs, max_new_tokens = 500, use_cache = True, eos_token_id = stop_token_id)
    model_generations = tokenizer.batch_decode(outputs)
    results.append(parse_generation(model_generations[0]))
    # print(parse_generation(model_generations[0]))
    # print('#######################################')

import pandas as pd

df = pd.DataFrame(results)
df.to_csv("phi3_mini_disjoint_results_step_250.csv")

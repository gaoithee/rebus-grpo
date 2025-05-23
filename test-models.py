from unsloth import FastLanguageModel

max_seq_length = 1248 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

model_type = "phi-3" # llama, phi-3, gemma

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "saracandu/phi3-mini-rebus-solver-adapter-grpo", 
    max_seq_length = max_seq_length,
    dtype = dtype,
)
FastLanguageModel.for_inference(model)

from datasets import load_dataset

eval_dataset = load_dataset('gsarti/eureka-rebus', 'llm_sft', data_files=["id_test.jsonl", "ood_test.jsonl"], split = "train")

import re

regex_word_guess = '- \[.* = (.*)'
regex_firstpass = 'Prima lettura: (.*)'
regex_solution_word = "\d+ = (.*)"
regex_solution = "Soluzione: (.*)"

def parse_generation(ex_idx, ex):
    try:
        word_guesses = ";".join(re.findall(regex_word_guess, ex))
    except:
        word_guesses = ""
    try:
        first_pass = re.findall(regex_firstpass, ex)[0]
    except:
        first_pass = ""
    try:
        solution_words = ";".join(re.findall(regex_solution_word, ex))
    except:
        solution_words = ""
    try:
        solution = re.findall(regex_solution, ex)[0]
    except:
        solution = ""
    return {
        "idx": ex_idx,
        "word_guesses": word_guesses,
        "first_pass": first_pass,
        "solution_words": solution_words,
        "solution": solution,
    }

import pandas as pd
from tqdm import tqdm

# Create gold parsed outputs

results = []
for ex_idx, ex in tqdm(enumerate(eval_dataset), total=len(eval_dataset)):
    gold_output = ex["conversations"][1]["value"]
    parsed_output = parse_generation(ex_idx, gold_output)
    results.append(parsed_output)

df = pd.DataFrame(results)
df.to_csv("test_gold_id_ood.csv")

from tqdm import tqdm

if model_type == "llama":
    tokenizer.padding_side = "right"
elif model_type in ("phi-3", "gemma"):
    tokenizer.padding_side = "left"

results = []
batch_size = 134
for i in tqdm(range(0, len(eval_dataset), batch_size), total=len(eval_dataset)//batch_size):
    batch = eval_dataset[i:i+batch_size]

    if model_type == "llama":
        input = [[{"role": "user", "content": example[0]["value"]}] for example in batch["conversations"]]
    elif model_type == "phi-3":
        input = [[item[0]] for item in batch["conversations"]]

    inputs = tokenizer.apply_chat_template(
        input,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        padding=True,
        truncation=True,
        return_dict=True
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True, eos_token_id = stop_token_id)

    model_generations = tokenizer.batch_decode(outputs)
    for ex_idx, ex in enumerate(model_generations):
        out_dic = parse_generation(ex_idx + i, ex)
        if i == 0 and ex_idx <= 5:
            print(ex)
            print(out_dic)
        results.append(out_dic)

import pandas as pd

df = pd.DataFrame(results)
df.to_csv("phi3_mini_results_step_750.csv")



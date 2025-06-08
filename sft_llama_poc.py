import os
token = os.getenv("HF_TOKEN")

from unsloth import FastLanguageModel
import torch

max_seq_length = 1248 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = False, # True or "unsloth" for very long context
    random_state = 4249,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


from datasets import load_dataset, Dataset
import pandas as pd

template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{human}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{gpt}<|eot_id|>"""

def formatting_prompts_func(examples):
    texts = [
        template.format(human=human, gpt=gpt)
        for human, gpt in zip(examples["merged_prompt"], examples["answer"])
    ]
    return {"text": texts}

train_dataset = load_dataset('saracandu/eureka-rebus-primalettura', split = "train")
test_dataset = load_dataset('saracandu/eureka-rebus-primalettura', split = "test")

train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
test_dataset = test_dataset.map(formatting_prompts_func, batched = True)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    data_collator=collator,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 100,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        output_dir = "outputs",
        lr_scheduler_type = "linear",
        seed = 42,
        save_steps = 500,
        save_total_limit = 10,
        eval_strategy = "steps",
        eval_steps = 500,
        push_to_hub = True,
        hub_model_id = "llama-3.1-8b-primeletture",
        report_to="wandb"
    ),
)


trainer_stats = trainer.train()

# Save merged model in FP16
model.push_to_hub_merged("llama-3.1-8b-primeletture-fp16", tokenizer, save_method = "merged_16bit")

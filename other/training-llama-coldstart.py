from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import wandb

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
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

dataset = load_dataset('saracandu/rebus-grpo-reasoning', split = "train")
# print(dataset)

# Trasforma la struttura per il fine-tuning
def format_example(example):
    
    # Struttura compatibile con fine-tuning
    full_text = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|start_header_id|>assistant<|end_header_id|>
    {assistant_output}<|eot_id|>
    """
    
    text = full_text.format(
        system_prompt = example["system-prompt"],
        user_prompt = example["prompt"],
        assistant_output = example["answer"]
    )
    return {"text": full_text}

# Applica la trasformazione
formatted_ds = dataset.map(format_example)
print(formatted_ds)

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# response_template = "<|assistant|>"
# collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_ds,
    eval_dataset = None,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1, # Tested on a 24GB RTX 3090 with 4bit quantization
        gradient_accumulation_steps = 1,
        warmup_steps = 10,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 50,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "cold-start-llama",
        num_train_epochs = 10,
        save_steps = 50,
        save_total_limit = None,
        eval_strategy = 'no',
        eval_steps = None,
        push_to_hub = False,
        hub_model_id = None,
        report_to="wandb"
    ),
)

wandb.init(project="cold-start-llama")
print("Training starts")
trainer_stats = trainer.train()
print("Training ends")


model.push_to_hub_merged("llama-3.1-8b-rebus-solver-coldstart", tokenizer, save_method = "merged_16bit")

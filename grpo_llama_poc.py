from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
from utils import check_soluzione, check_prima_lettura, check_chiave_risolutiva
import wandb

import os
token = os.getenv("HF_TOKEN")
import torch

max_seq_length = 1248 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "saracandu/llama-3.1-8b-primeletture",
    max_seq_length = max_seq_length,
    dtype = dtype,
    fast_inference = True,
    load_in_4bit = load_in_4bit,
    revision = "ed39ab1d51baa99e1ee8467d2077dd224ec4009c",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = False, # True or "unsloth" for very long context
    random_state = 4249,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

model_type = "llama"
stop_token_id = model.config.eos_token_id
if model_type == "gemma":
    stop_token = "<|eot_id|>"
    stop_token_id = tokenizer.encode(stop_token)[0]
if model_type == "llama":
    tokenizer.padding_side = "right"
elif model_type in ("phi-3", "gemma"):
    tokenizer.padding_side = "left"

eval_dataset = load_dataset('saracandu/eureka-rebus-primalettura', data_files = ['train.csv'], split="train")

eval_dataset = eval_dataset.select(range(13500))
eval_dataset = eval_dataset.remove_columns(['Unnamed: 0'])
eval_dataset = eval_dataset.map(lambda x: {  # type: ignore
    'prompt': [
        {'role': 'system', 'content': x['system']},
        {'role': 'user', 'content': x['prompt']}
    ],
    'answer': x['answer']
})

training_args = GRPOConfig(
    learning_rate=5e-6, # può essere sensato tenerlo piccolo perché è già stato fine-tuned
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit", # risparmia in memoria & aumenta la velocità
    logging_steps=50,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=256,
    max_completion_length=500,
    num_train_epochs = 1, # Set to 1 for a full training run
    save_steps=50,
    max_grad_norm=0.1,
    report_to = ["wandb"],
    output_dir="primeletture-llama",
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[check_soluzione, check_prima_lettura, check_chiave_risolutiva],
    args=training_args,
    train_dataset=eval_dataset,
)

trainer.train()

# -*- coding: utf-8 -*-

# import unsloth
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
# attenzione che qua ti printa tutto!
# se vuoi vedere qualcosa di sensato, metti batch_size = 1 e gradient_acc_steps = 1!
from reward_funcs_new import check_word_guesses, check_first_pass, check_solution_words, check_solution
import wandb


max_seq_length = 1248 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model_type = "phi-3" # llama, phi-3, gemma
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "gsarti/phi3-mini-rebus-solver-adapters",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = True, # altrimenti crasha 
    fast_inference = True, 
    enable_prefix_caching = False,  # obbligati
    disable_sliding_window = False  # obbligati
)


stop_token_id = model.config.eos_token_id
if model_type == "gemma":
    stop_token = "<|eot_id|>"
    stop_token_id = tokenizer.encode(stop_token)[0]

if model_type == "llama":
    tokenizer.padding_side = "right"
elif model_type in ("phi-3", "gemma"):
    tokenizer.padding_side = "left"



eval_dataset = load_dataset('saracandu/eureka-rebus-grpo', data_files = ['train.csv'], split="train")
eval_dataset = eval_dataset.select(range(13500)) # potrei rimettere tutto il dataset una volta che abbiamo una run decente
eval_dataset = eval_dataset.remove_columns(["Unnamed: 0"]) # pasticcio mio, correggerò

eval_dataset = eval_dataset.map(lambda x: { 
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
    per_device_train_batch_size=12,
    gradient_accumulation_steps=1,
    num_generations=4,  # 6 mi sembrava troppo
    # 12 * 6 -> 72 batch_size ( -> 750 steps)
    max_prompt_length=256,
    max_completion_length=500, # uguale alle tue richieste lato SFT
    num_train_epochs = 1, # non voglio che overfitti
    save_steps=50,
    max_grad_norm=0.1,
    report_to = ["wandb"],
    output_dir="GRPO-phi-new",
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[check_word_guesses, check_first_pass, check_solution_words, check_solution],
    args=training_args,
    train_dataset=eval_dataset,
)

wandb.init(project="phi-GRPO-new")
print("Training begins...")
trainer.train()
print("Training ends!")

# merged_model = trainer.model.merge_and_unload()
# merged_model.push_to_hub(
#     "phi3-mini-rebus-solver-grpo-new", private=False, tags=["GRPO", "phi3"]
# )
# tokenizer.push_to_hub("phi3-mini-rebus-solver-grpo-new")



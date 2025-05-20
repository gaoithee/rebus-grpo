from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from tqdm import tqdm
from reward_funcs import exact_match_solution, perc_correct_words_solution, words_letters_match_primalet, perc_correct_words_defres
import wandb
wandb.login(key="5a69225ea1d050c9c21f67c2db85febf61fa8fb1")


max_seq_length = 1248 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model_type = "phi-3" # llama, phi-3, gemma
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "gsarti/phi3-mini-rebus-solver-adapters", # MODEL OR ADAPTER FOLDER
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
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



training_args = GRPOConfig(
    learning_rate=5e-6, # può essere sensato tenerlo piccolo perché è già stato fine-tuned
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit", # risparmia in memoria & aumenta la velocità
    logging_steps=50,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Increase to 4 for smoother training
    num_generations=6,  # Decrease if out of memory
    max_prompt_length=256,
    max_completion_length=500,
    num_train_epochs = 3, # Set to 1 for a full training run
    save_steps=1000,
    max_grad_norm=0.1,
    report_to = ["wandb"],
    output_dir="GRPO-phi",
)



trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[exact_match_solution, perc_correct_words_solution, words_letters_match_primalet, perc_correct_words_defres],
    args=training_args,
    train_dataset=eval_dataset,
)

wandb.init(project="phi-GRPO")
trainer.train()


merged_model = trainer.model.merge_and_unload()
merged_model.push_to_hub(
    "phi3-mini-rebus-solver-adapter-grpo", private=False, tags=["GRPO", "phi3"]
)
tokenizer.push_to_hub("phi3-mini-rebus-solver-adapter-grpo")



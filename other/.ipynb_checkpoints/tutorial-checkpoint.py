from huggingface_hub import login
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import wandb


login(token=access_token)

dataset = load_dataset("mlabonne/smoltldr")

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

lora_config = LoraConfig(
    task_type = "CAUSAL_LM",
    r = 16, # rank of the surrogate matrices!
    lora_alpha = 32, # scale factor controlling the impact of the modifications
    target_modules = "all-linear" # applied to all linear transformations in the model
)
model = get_peft_model(model, lora_config)

ideal_length = 50
def reward_function(completions, **kwargs):
  return [-abs(ideal_length - len(completion)) for completion in completions]

training_args = GRPOConfig(
    output_dir = "GRPO",
    learning_rate = 2e-5,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,
    max_prompt_length = 512, 
    max_completion_length = 96,
    num_generations = 8, # i.e. number of competitive completions considered during optimization
    optim = "adamw_8bit",
    num_train_epochs = 1,
    bf16 = True, 
    report_to = ["wandb"],
    remove_unused_columns = False, 
    logging_steps = 1
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_function],
    args=training_args,
    train_dataset=dataset["train"],
)

wandb.init(project="GRPO")
trainer.train()


merged_model = trainer.model.merge_and_unload()
merged_model.push_to_hub(
    "SmolGRPO-135M", private=False, tags=["GRPO", "Reasoning-Course"], token=access_token
)


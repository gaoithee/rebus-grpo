from unsloth import FastLanguageModel
import torch

max_seq_length = 1248 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct-v0-bnb-4bit",
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





from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-3",
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

dataset = load_dataset('gsarti/eureka-rebus', 'llm_sft', data_files=["train.jsonl"], split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True)




eval_dataset = load_dataset('gsarti/eureka-rebus', 'llm_sft', data_files=["id_test.jsonl", "ood_test.jsonl"], split = "train")
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)


from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

response_template = "<|assistant|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    data_collator=collator,
    args = TrainingArguments(
        per_device_train_batch_size = 32, # Tested on a 24GB RTX 3090 with 4bit quantization
        gradient_accumulation_steps = 1,
        warmup_steps = 100,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 150,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
        num_train_epochs = 1,
        save_steps = 100,
        save_total_limit = 3,
        eval_strategy = "steps",
        eval_steps = 500,
        push_to_hub = True,
        hub_model_id = "phi3-mini-rebus-solver-coldstart-retrained",
        report_to="wandb",
        max_steps=500
    ),
)

trainer_stats = trainer.train()

model.push_to_hub_merged("phi3-mini-rebus-solver-coldstart-retrained", tokenizer, save_method = "merged_16bit")


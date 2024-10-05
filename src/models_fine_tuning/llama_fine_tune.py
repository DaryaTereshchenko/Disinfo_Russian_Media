import torch
from trl import SFTTrainer
from datasets import load_from_disk
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
import os
from dotenv import load_dotenv
load_dotenv()

huggingface_model_name = "dariast/Meta-Llama-3.1-8B-4bit-python"

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized

    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None # And LoftQ
)

chat_template = """{SYSTEM}
USER: {INPUT}
ASSISTANT: {OUTPUT}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["SYSTEM"]
    inputs = examples["INPUT"]
    outputs = examples["OUTPUT"]
    texts = []
    
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Adding a check for None values to avoid potential issues
        if instruction is None or input_text is None or output is None:
            continue
            
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = chat_template.format(SYSTEM=instruction, INPUT=input_text, OUTPUT=output) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}

# Load the dataset
dataset = load_from_disk("/home/dariast/Disinfo_Russian_Media/data/fine_tuning_dataset")

# Access train and test splits
train_ds = dataset['train']
test_ds = dataset['test']

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 4, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# 6. Saving
model.save_pretrained("llama_fine_tuned") # Local saving
tokenizer.save_pretrained("llama_fine_tuned")
model.push_to_hub(huggingface_model_name, token = os.getenv("HF_TOKEN")) 
tokenizer.push_to_hub(huggingface_model_name, token = os.getenv("HF_TOKEN"))
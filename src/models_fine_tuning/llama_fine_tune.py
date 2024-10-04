from unsloth import FastLanguageModel
from unsloth import apply_chat_template, standardize_sharegpt
import torch
import pandas as pd

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
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

chat_template = """{SYSTEM}
USER: {INPUT}
ASSISTANT: {OUTPUT}"""

# read the data in json format
df = pd.read_csv(open("/home/dariast/Disinfo_Russian_Media/data/zero_shot_fine_tuning.csv", "r", encoding='utf-8'))
df_renamed = df.rename(columns={"text": "INPUT", "class": "OUTPUT"})
dataset = df_renamed.drop(columns=["article_id"])

dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    default_system_message = "You are a helpful assistant who is fluent in Russian and can classify text accurately."
)
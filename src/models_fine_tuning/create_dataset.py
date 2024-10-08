import pandas as pd
import os 
from convert_prompts import read_csv, sample_data, select_columns, save_to_csv, save_to_json, read_prompts_from_txt
from typing import List, Dict
from datasets import load_dataset
from unsloth import FastLanguageModel
from datasets import DatasetDict

def create_training_dataset(file_path: str, output_csv_path: str, prompt_path: str) -> None:
    df = read_csv(file_path)
    selected_df = select_columns(df, ['class', 'text'])
    
    # Read the prompts from the text file
    system_prompt, user = read_prompts_from_txt(prompt_path)
    user_prompt_template, assistant_prompt = user.split("assistant:")[0], user.split("assistant:")[1]
    system_prompts = []
    user_prompts = []
    assistant_prompts = []
    for _, row in selected_df.iterrows():
        text = row['text']
        label = row['class']
        system_prompts.append(system_prompt)
        user_prompts.append(user_prompt_template.format(text=text))
        assistant_prompts.append(assistant_prompt.format(label=label))
    # Create a new DataFrame with the prompts
    new_df = pd.DataFrame({'SYSTEM': system_prompts, 'INPUT': user_prompts, 'OUTPUT': assistant_prompts})
    csv_name = "fine_tuning_subset.csv"
    save_to_csv(new_df, os.path.join(output_csv_path, csv_name))

# Template for fine-tuning
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT}<|eot_id|>"""

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

_, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)
EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# Formatting function to apply to the dataset
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
        text = chat_template.format(SYSTEM=instruction, INPUT=input_text, OUTPUT=output)
        texts.append(text)
    return {"text": texts}

def main(path_to_csv: str, output_path: str):

    # Load dataset
    ds = load_dataset('csv', data_files=path_to_csv, split='train')

    # Split dataset into train and test sets (e.g., 80% train, 20% test)
    split_datasets = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_datasets['train']
    test_ds = split_datasets['test']

    # Apply the formatting function to the train and test datasets
    train_dataset = train_ds.map(formatting_prompts_func, batched=True)
    test_dataset = test_ds.map(formatting_prompts_func, batched=True)
    # Combine train and test datasets into a DatasetDict
    combined_dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    # Save the combined dataset to disk
    combined_dataset.save_to_disk(output_path)

if __name__ == "__main__":
    create_training_dataset('./data/zero_shot_fine_tuning.csv', './prompts/prompts_json', './prompts/templates/zero_shot_fine_tuning.txt')
    main('./data/fine_tuning_subset.csv', './data/fine_tuning_dataset')
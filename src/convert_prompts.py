import os 
import pandas as pd
import json
from typing import List, Dict, Optional

def read_csv(file_path: str) -> pd.DataFrame:
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)

def sample_data(df: pd.DataFrame, num_rows: int = 50, random_seed: int = 42) -> pd.DataFrame:
    """Takes a random sample of data from a DataFrame."""
    return df.sample(n=num_rows, random_state=random_seed)

def select_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Selects specific columns from the DataFrame."""
    return df[columns]

def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """Saves the DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)

def read_prompts_from_txt(file_path: str):
    """Reads the system and user prompts from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        # Split text based on user and system prompts
        content = file.read()
        system_prompt = content.split("system:")[-1].strip()
        user_prompt_format = content.split("user:")[-1].strip()

        return system_prompt, user_prompt_format

def create_json_template(data: pd.DataFrame, user_prompt_template: str, system_prompt: str, template_type: str = "zero-shot") -> List[Dict[str, str]]:
    """
    Creates a JSON structure based on the template type (currently supports 'zero-shot').
    """
    if template_type == "zero-shot":
        json_data = []
        json_data.append({"role": "system", "content": system_prompt})
        for _, row in data.iterrows():
            text = row['limited_text']
            add_text = user_prompt_template.format(text=text)
            json_data.append({"role": "user", "content": add_text})
        return json_data
    # Placeholder for future few-shot template addition
    elif template_type == "few-shot":
        # Logic for few-shot template will be added here later
        pass

def save_to_json(json_data: List[Dict[str, str]], output_path: str) -> None:
    """Saves the JSON data to a file."""
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

# Main function to handle the entire process
def process_data(file_path: str, output_csv_path: str, output_json_path: str, template_type: str = "zero-shot") -> None:
    df = read_csv(file_path)
    sampled_df = sample_data(df)
    selected_df = select_columns(sampled_df, ['article_id', 'class', 'limited_text'])

    # Read the prompts from the text file
    zero_shot_prompts_path = os.path.join('.\prompts', 'zeroshot_ver_01_template.txt')
    system_prompt_template, user_prompt_template = read_prompts_from_txt(zero_shot_prompts_path)

    # Save the selected data to CSV
    save_to_csv(selected_df, output_csv_path)

    # Create and save the JSON data based on the specified template type
    json_data = create_json_template(selected_df, user_prompt_template, system_prompt_template, template_type)
    save_to_json(json_data, output_json_path)
    
    print(f"Data saved to {output_csv_path} and {output_json_path}")


if __name__ == "__main__":
    file_path = os.path.join('.\data', 'cleaned_data.csv')
    
    output_csv_path = os.path.join('.\data', 'sampled_data_zero_shot.csv')
    output_json_path = os.path.join('.\prompts\prompts_json', 'zero_shot.json')
    
    process_data(file_path, output_csv_path, output_json_path, template_type="zero-shot")
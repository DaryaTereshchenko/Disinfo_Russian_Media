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

def create_json_template(data, template_type="zero-shot"):
    """
    Creates a JSON structure based on the template type (currently supports 'zero-shot').
    """
    if template_type == "zero-shot":
        json_data = []
        for _, row in data.iterrows():
            json_data.append({"role": "user", "content": row['limited_text']})
        return json_data
    # Placeholder for future few-shot template addition
    elif template_type == "few-shot":
        # Logic for few-shot template will be added here later
        pass

def save_to_json(json_data, output_path):
    """Saves the JSON data to a file."""
    with open(output_path, 'w') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

# Main function to handle the entire process
def process_data(file_path, output_csv_path, output_json_path, template_type="zero-shot"):
    df = read_csv(file_path)
    sampled_df = sample_data(df)
    selected_df = select_columns(sampled_df, ['article_id', 'class', 'limited_text'])
    
    # Save the selected data to CSV
    save_to_csv(selected_df, output_csv_path)
    
    # Create and save the JSON data based on the specified template type
    json_data = create_json_template(selected_df, template_type)
    save_to_json(json_data, output_json_path)
    
    print(f"Data saved to {output_csv_path} and {output_json_path}")

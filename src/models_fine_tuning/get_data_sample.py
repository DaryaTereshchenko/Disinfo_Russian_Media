import pandas as pd
import os 
from src.convert_prompts import read_csv, sample_data, select_columns, save_to_csv

def create_data_for_training(path:str, num_samples:int, random_seed:int, output_csv_path:str):
    df = read_csv(path)
    sampled_df = sample_data(df, num_samples, random_seed)
    selected_df = select_columns(sampled_df, ['article_id', 'class', 'text'])
    save_to_csv(selected_df, output_csv_path)

if __name__ == '__main__':
    path = os.path.join("./data", "cleaned_data.csv")
    num_samples = 300
    random_seed = 42
    output_csv_path = os.path.join("./data", "sampled_data_fine_tining.csv")
    create_data_for_training(path, num_samples, random_seed, output_csv_path)
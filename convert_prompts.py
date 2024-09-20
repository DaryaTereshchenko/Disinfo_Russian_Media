import os 
import json 
import pandas as pd

# Read the the file from 'prompts' folder, rewrite it according to the format of the prompts
#  Save the new file in the 'prompts' folder in the json format

def read_text_label(file_path_text: str) -> str:
    df = pd.read_csv(file_path_text)
    text = df['text'].values
    pass

def convert_prompts_zeroshot(file_path_prompt: str, data: str) -> dict:
    pass

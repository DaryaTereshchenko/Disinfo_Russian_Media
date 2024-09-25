import openai
import logging
from tqdm import tqdm
from utils import plot_count_and_normalized_confusion_matrix, log_metrics_and_confusion_matrices_wandb
import pandas as pd
import wandb
import os
import json
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv('API_KEY')
base_url = os.getenv('BASE_URL')
WANDB_PROJECT_NAME = 'llm-classification'

client = openai.OpenAI(
    base_url=base_url,
    api_key=api_key,
)

def run_experiment(df: pd.DataFrame, experiment_name:str, dataset_name: str, model_name: str, prompts: str, task_name:str) -> pd.DataFrame:
    # Start logging
    wandb.init(
        # set the wandb project where this run will be logged
        project=WANDB_PROJECT_NAME,
        name=experiment_name,
        # track hyperparameters and run metadata
        config = {
            "model": model_name,
            "dataset": dataset_name
        }
    )
    
    # Log prompts
    prompts_artifact = wandb.Artifact('prompts', type='prompts')

    # Read json file and save the first prompt as system prompt
    with open(prompts, 'r', encoding='utf-8') as file:
        prompts = json.load(file)

    system_prompt = prompts[0]['content']
    with prompts_artifact.new_file('system_prompt.txt', mode='w', encoding='utf-8') as f:
        f.write(system_prompt)
    # Save other prompts as user prompts
    user_prompt_format = "\n\n".join([prompt['content'] for prompt in prompts[1:]])
    with prompts_artifact.new_file('user_prompt_format.txt', mode='w', encoding='utf-8') as f:
        f.write(user_prompt_format)
    wandb.run.log_artifact(prompts_artifact)
    
    # Evaluate an LLM on the dataset
    outputs = []
    system_prompt = prompts[0]['content']

    for _, prompt in enumerate(tqdm(prompts[1:],  desc="Processing prompts")):
        text = prompt['content']

        response = client.chat.completions.create(
            model=model_name,
            temperature=0, # make the output deterministic
            max_tokens=4, # make sure the model generates the expected output
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        )
        
        full_response = response.choices[0].message.content
        outputs.append(full_response)
    
    processed_outputs = []
    for output in outputs:
        if output == 'disinformation':
            processed_outputs.append('disinformation')
        elif output == 'trust' or 'trustworthy':
            processed_outputs.append('trustworthy')
        else:
            processed_outputs.append('unidentified')
    print(len(df), len(processed_outputs))
    # # Add the outputs to the dataframe
    # df[f"predicted_{task_name}"] = processed_outputs

    # print(df)
    
    # Log the dataset output to wandb
    # predictions_artifact = wandb.Artifact('predictions', type='outputs')
    # with predictions_artifact.new_file('predictions.csv', mode='w', encoding='utf-8') as f:
    #     df.to_csv(f)
    # wandb.run.log_artifact(predictions_artifact)


    # # Measure performance 
    # try:    
    #     y_true = df['class']
    #     y_pred = df[f"predicted_{task_name}"]
    
    #     cm_plot, classification_report, metrics = plot_count_and_normalized_confusion_matrix(y_true=y_true, y_pred=y_pred)
        
    #     log_metrics_and_confusion_matrices_wandb(cm_plot=cm_plot, classification_report=classification_report, metrics=metrics, task_name='zero-shot')

    # except Exception as e:
    #     print('Error computing metrics: ', e)

    # # Finish logging
    # wandb.finish()

    return df
    
if __name__ == '__main__':
    # Load the dataset
    data_path = os.path.join("./data", "sampled_data_zero_shot.csv")
    path_prompts = os.path.join("./prompts/prompts_json", "zero_shot.json")

    df = pd.read_csv(data_path)
    models = ["mistral-nemo:12b", "gemma2:9b", "ollama run llama3.1:8b"]
    result = run_experiment(df, 'zero-shot', 'sampled_data_zero_shot', "mistral-nemo:12b", path_prompts, 'zero_shot')

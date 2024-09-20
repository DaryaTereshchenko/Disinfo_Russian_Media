import openai
import logging
from tqdm import tqdm
from utils import process_output, plot_count_and_normalized_confusion_matrix, log_metrics_and_confusion_matrices_wandb
import pandas as pd
import wandb
import os
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

def run_experiment(df: pd.DataFrame, system_prompt: str, user_prompt_format: str, 
                   experiment_name:str, dataset_name: str, model_name: str,
                   subset_output_fun: callable = lambda x: x):
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
    with prompts_artifact.new_file('system_prompt.txt', mode='w') as f:
        f.write(system_prompt)
    with prompts_artifact.new_file('user_prompt_format.txt', mode='w') as f:
        f.write(user_prompt_format)
    wandb.run.log_artifact(prompts_artifact)
    
    # Evaluate ChatGPT on the dataset
    outputs = []
    response = client.chat.completions.create(
        model="mistral-nemo:12b",
        temperature=0.7,
        max_tokens=256,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who is fluent in Russian."},
            {"role": "user", "content": "Напиши мне что-нибудь по-русски."},
        ],
    )
        
        # Process the output
    relevant, framing = process_output(subset_output_fun(chat_gpt_message_output))
        
        # Add to the list
    outputs.append(
        {
            "relevant": relevant, 
            "frame": framing,
            "full_output_chatgpt": chat_gpt_message_output
        }
    )
    
    # Join the outputs to the original dataframe
    outputs_df = pd.DataFrame(outputs, index=df.index)
    df = df.join(outputs_df, rsuffix='_pred')
    
    # Log the dataset output to wandb
    predictions_artifact = wandb.Artifact('predictions', type='outputs')
    with predictions_artifact.new_file('predictions.csv', mode='w') as f:
        df.to_csv(f)
    wandb.run.log_artifact(predictions_artifact)
    
    # Measure performance 
    try:
        # Relevance task
        y_true_relevance = df['relevant'].map(int)
        y_pred_relevance = outputs_df['relevant'].fillna(-1).map(int)
        cm_plot, classification_report, metrics  =  plot_count_and_normalized_confusion_matrix(y_true=y_true_relevance, y_pred=y_pred_relevance, labels=[1, 0], display_labels=['Relevant', 'Irrelevant'], custom_display_labels=True)
        log_metrics_and_confusion_matrices_wandb(cm_plot=cm_plot, classification_report=classification_report, metrics=metrics, task_name='relevance')
    
        # Problem solution framing task
        y_true_frame = df.loc[~df['frame'].isna(), 'frame'].map(
            {'Neither': 0, 'Solution': 1, 'Problem': 2})
        y_pred_frame = outputs_df.loc[~df['frame'].isna(), 'frame'].fillna(-1).map(int)
    
        cm_plot, classification_report, metrics = plot_count_and_normalized_confusion_matrix(y_true=y_true_frame, y_pred=y_pred_frame, labels=[0, 1, 2, 3], display_labels=['Neither', 'Solution', 'Problem', 'Both'], custom_display_labels=True)
        log_metrics_and_confusion_matrices_wandb(cm_plot=cm_plot, classification_report=classification_report, metrics=metrics, task_name='framing')
        wandb.finish()
        
    except Exception as e:
        print('Error computing metrics: ', e)
    
    return df
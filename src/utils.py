from typing import Any
import regex as re 
import numpy as np
import json
import wandb
import logging
import pandas as pd
import pprint
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, classification_report, recall_score,
                             precision_score, f1_score)
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, filename='inference.log', format='%(asctime)s - %(levelname)s - %(message)s')
wandb.login()


# def extract_label_with_regex(input_string: str, pattern: str, group_to_return: int) -> float | Any:
#     # Use re.search to find the match
#     match = re.search(pattern, input_string)

#     if match:
#         # Extract the value of "relevance" from the match
#         value = str(match.group(group_to_return))
#         return value
#     else:
#         return np.nan
    
# extract_label = lambda x: extract_label_with_regex(x, pattern=r'"label"\s*:\s*w+', group_to_return=2)

# def process_output(llm_message_output: str) -> list | Any:
#     try:
#         output = json.loads(llm_message_output)
#         get_label = 'label' if 'label' in output else None
        
#         output_classes = output[get_label] if get_label else np.nan

#     except json.JSONDecodeError as e:
#         print(f'Error parsing JSON: {e}')
#         print(f'LLM Output: {llm_message_output}')
#         output_classes = extract_label(llm_message_output)
    
#     except KeyError as e:
#         print(f'JSON does not contain the "label" in keys: {e}')
#         print(f'JSON: {output}')
#         output_classes = np.nan
        
#     except ValueError as e:
#         print(f'Output classes could not be parsed as strings: {e}')
#         print(f'label: {output["label"]}')
        
#         output_classes = output['label']
    
#     except Exception as e:
#         print(f'Unknown error: {e}')
#         print(f'ChatGPT output: {llm_message_output}')
#         output_classes = np.nan

#     return output_classes

def log_metrics_and_confusion_matrices_wandb(cm_plot, classification_report, metrics, task_name: str):
    # Log metrics
    for metric_name, metric_value in metrics.items():
        wandb.log({f'{metric_name}_{task_name}': metric_value})

    # Log the confusion matrix matplotlib figure
    wandb.log({f'confusion_matrix [{task_name}]': wandb.Image(cm_plot)})
    
    # Log the classification report as an artifact
    classification_report = (pd.DataFrame({k: v for k, v in classification_report.items() if k != 'accuracy'})
                             .transpose().reset_index())
    
    wandb.log({f'classification_report [{task_name}]': wandb.Table(
        dataframe=classification_report)})
    
    classification_report_artifact = wandb.Artifact(
        f'classification_report_{task_name}', type='classification_report')
    
    with classification_report_artifact.new_file(f'classification_report_{task_name}.txt', mode='w') as f:
        f.write(pprint.pformat(classification_report))
    wandb.run.log_artifact(classification_report_artifact)


default_metrics = {
    'accuracy': accuracy_score,
    'recall': lambda y_t, y_p: recall_score(y_t, y_p, zero_division=np.nan, average='micro'),
    'precision': lambda y_t, y_p: precision_score(y_t, y_p, zero_division=np.nan, average='micro'),
    'f1': lambda y_t, y_p: f1_score(y_t, y_p, zero_division=np.nan, average='micro')
}

def plot_count_and_normalized_confusion_matrix(y_true, y_pred, xticks_rotation='horizontal',
                                               metrics: dict = default_metrics, custom_display_labels: bool = False):
    # Print classification report
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    pprint.pprint(cls_report)

    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Remove labels and display_labels not present in y_true
    # if not custom_display_labels:
    #     display_labels = [label for label in display_labels if label in y_true.unique()]
    #     labels = [label for label in labels if label in y_true.unique()]

    # Plot count confusion matrix
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    cm_disp.plot(ax=ax1, xticks_rotation=xticks_rotation)
    ax1.set_title('Count Confusion Matrix')

    # Plot normalized confusion matrix
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true')
    cm_disp.plot(ax=ax2, xticks_rotation=xticks_rotation)
    ax2.set_title('Normalized Confusion Matrix')

    # Show plot
    plt.show()
    plt.close()

    # Calculate metrics
    metrics = {metric_name: metric_func(y_true, y_pred) for metric_name, metric_func in metrics.items()}

    return fig, cls_report, metrics
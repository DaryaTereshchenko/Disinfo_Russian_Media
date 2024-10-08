from typing import Any
import numpy as np
import wandb
import regex as re
import logging
import pandas as pd
import pprint
import os
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, classification_report, recall_score,
                             precision_score, f1_score)
import matplotlib.pyplot as plt
os.environ["WANDB_CONSOLE"] = "off"
logging.basicConfig(level=logging.INFO, filename='inference.log', format='%(asctime)s - %(levelname)s - %(message)s')
# wandb.login()

def log_metrics_and_confusion_matrices_wandb(cm_plot_path, classification_report, metrics, task_name: str):
    # Log metrics
    for metric_name, metric_value in metrics.items():
        wandb.log({f'{metric_name}_{task_name}': metric_value})

    # Log the confusion matrix matplotlib figure
    wandb.log({f'confusion_matrix_{task_name}': wandb.Image(cm_plot_path)})
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
                                               metrics: dict = default_metrics, save_path=None,  figure_title=None):
    # Print classification report
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    pprint.pprint(cls_report)

    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot count confusion matrix
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax1)
    ax1.set_title('Count Confusion Matrix')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=xticks_rotation)

    # Plot normalized confusion matrix
    cm_disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='true', ax=ax2)
    ax2.set_title('Normalized Confusion Matrix')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=xticks_rotation)
    
    if figure_title:
        fig.suptitle(figure_title, fontsize=16)
        # Create a valid filename from the figure title
        sanitized_title = re.sub(r'[^a-zA-Z0-9]', '_', figure_title)
        save_filename = f'confusion_matrix_{sanitized_title}.png'
    else:
        save_filename = 'confusion_matrix.png'
    
    # Save plot if a save path is provided
    if save_path:
        print(f"Saving confusion matrix to {save_path}")
        full_save_path = os.path.join(save_path, save_filename)
        plt.savefig(full_save_path)  # Save the figure
        print(f"Confusion matrix saved to {full_save_path}")
    else:
        os.makedirs("./img", exist_ok=True)
        full_save_path = os.path.join("./img", save_filename)
        plt.savefig(full_save_path)

    # # Show plot
    plt.show()
    plt.close()

    # Calculate metrics
    metrics = {metric_name: metric_func(y_true, y_pred) for metric_name, metric_func in metrics.items()}

    return full_save_path, cls_report, metrics

def process_output(input_text: str) -> str:
    matches = re.findall(r"\b(disinformation|trustworthy|dis|trust|Dis|Trust|Disinformation|Trustworthy)\b", input_text)
    if matches:
        matched = matches[0]
        if matched.lower() == 'disinformation' or matched.lower() == 'dis':
            return 'disinformation'
        if matched.lower() == 'trustworthy' or matched.lower() == 'trust':
            return 'trustworthy'
    return 'undefined'
from datasets import load_from_disk
from utils import process_output
from unsloth import FastLanguageModel
import re

# Load the dataset
dataset = load_from_disk("fewshot_dataset")
# Access train and test splits
train_ds = dataset['train']
test_ds = dataset['test']

max_seq_length = 2048 
dtype = None 
load_in_4bit = False 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "mistral_fine_tuned",
    max_seq_length = max_seq_length,
    dtype = dtype, 
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

def substitute_last_part_of_prompt(input_string):
    result = re.sub(r'\n{3}"Class": .*?<\|eot_id\|>', '', input_string)
    return result
test_prompts = [substitute_last_part_of_prompt(prompt) for prompt in train_ds['text']]

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

predictions = []
# Iterate through all test instances
for i, prompt in enumerate(test_prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=10)
    predicted_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    predictions.append(predicted_text)

processed_outputs = []
for output in predictions:
    result = output.split('\n\nassistant\n\n')[-1]
    processed_outputs.append(process_output(result))

# Clean true labels 
true_labels = [process_output(label) for label in train_ds['OUTPUT']]


from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
# Calculate confusion matrix
cm = confusion_matrix(true_labels, processed_outputs, normalize='true')

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=set(true_labels))
disp.plot(ax=ax, cmap='viridis', values_format='.2f')
fig.savefig("cfm_fewshot_mistral.png")

y_true = true_labels
y_pred = processed_outputs
default_metrics = {
    'accuracy': accuracy_score,
    'recall': lambda y_t, y_p: recall_score(y_t, y_p, zero_division=1, average='macro'),
    'precision': lambda y_t, y_p: precision_score(y_t, y_p, zero_division=1, average='macro'),
    'f1': lambda y_t, y_p: f1_score(y_t, y_p, zero_division=1, average='macro')
}
cls_report = classification_report(y_true, y_pred, output_dict=True)
metrics = {metric_name: metric_func(y_true, y_pred) for metric_name, metric_func in default_metrics.items()}

# Save metrics
with open("metrics_fewshot_mistral.txt", "w") as f:
    f.write(f"Metrics for Few-shot learning with Mistral model:\n")
    for metric_name, metric_value in metrics.items():
        f.write(f"{metric_name}: {metric_value}\n")
    f.write(f"\nClassification report:\n")
    for label, metrics in cls_report.items():
        f.write(f"{label}: {metrics}\n")
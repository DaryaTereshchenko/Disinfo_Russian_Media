# Exploring the Efficacy of Large Language Models in Identifying Russian Disinformation

## Project Overview
*This study explores the effectiveness of large language models (LLMs) in detecting Russian-language disinformation, a relatively under-explored area compared to English-language disinformation research. We evaluate multilingual LLMs using zero-shot, few-shot, and chain-of-thought prompting techniques on Russian disinformation datasets. The models are then fine-tuned on an annotated corpus to enhance detection accuracy. Additionally, we examine the capacity of LLMs to generate disinformation in Russian and translate it into English, assessing both their potential in disinformation detection and the risk of misuse. Our findings contribute to the growing body of research on multilingual disinformation detection, offering insights into LLM performance across languages and highlighting key ethical considerations in the use of AI for both mitigating and propagating disinformation.*

## Key Objectives

- Assess LLMs' performance in identifying Russian disinformation.
- Fine-tune models to improve detection capabilities.
- Compare prompting strategies (zero-shot, few-shot, CoT) for effectiveness.

## Model Selection
The selected models include:

- Mistral-Nemo 12B: A fine-tuned version of Mistral-Nemo-Base-2407.
- Google Gemma 2 9B: Part of Google’s Gemini series, optimized for complex tasks.
- Meta Llama 3.1 8B Instruct: A versatile model known for efficiency in natural language tasks.

## Project Results
The confusion matrices of prompts tested on the inference task can be found in ```cfm_base_models```. The confusion matrices of the fine-tuned models testing can be found in ```cfm_fine_tuned_models```.

You can find the fine-tuned models in the following folder: ```fine_tuned_models```.
The models can be re-run using the script in ```src/models_fine_tuning/test_fine_tuned.ipynb```

## Usage Tips 

**NOTE: Before runing the code create a .env file in the root directory and add the following variables:**
```bash
- API_KEY=your_openai_api_key
- HF_TOKEN=your_huggingface_token
- WANDB_KEY=your_wandb_key
```

- You can run the fine-tuning notebooks using a GPU with VRAM >= 32GB.
- You can adjust the prompts in ```prompts/templates``` to test different templates, add examples or modify the CoT resoning.
- If you want to test existing prompts, run ```src/prompts_testing.py```.

## How to Run
1. Clone the repository:
```bash
git https://github.com/DaryaTereshchenko/Disinfo_Russian_Media
cd Disinfo_Russian_Media
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
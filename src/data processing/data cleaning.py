import pandas as pd
import re
import spacy
from tqdm import tqdm
import os 

# Load the spaCy Russian model
nlp = spacy.load("ru_core_news_sm")

class DataProcessor:
    def __init__(self, file_path: str):
        self.data = pd.read_csv(file_path)
    
    def process_data(self):
        # Discard rows with missing values
        self.data.dropna(subset=["text"], inplace=True)

        # Clean the text data
        self.data["text"] = self.data["text"].apply(self.clean_text)
    
    def clean_text(self, text):
        if isinstance(text, float) or text is None:
            return ""
        # Remove "Error retrieving content" messages
        text = re.sub(r"Error retrieving content: \d+ .+", "", text)
        
        # Remove date-time information (ISO 8601 format)
        text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\+\d{2}:\d{2}|Z)?", "", text)
        
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        
        # Remove text inside curly braces {{...}}
        text = re.sub(r"\{\{.*?\}\}", "", text)
        
        # Remove non-Cyrillic characters (keep punctuation and spaces)
        text = re.sub(r"[^\u0400-\u04FF\s.,!?]", "", text)
        
        # Remove extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        
        if len(text.split()) < 10:
            return ""
        
        return text


    def get_limited_token_data(self, max_tokens=8000):
        # Enable the use of progress_apply with pandas
        tqdm.pandas()
        
        # Tokenize Russian text data using spaCy's ru_core_news_sm model
        def get_first_tokens(text, max_tokens):
            tokens = [token.text for token in nlp(text)]
            return ' '.join(tokens[:max_tokens])  # Get the first `max_tokens` tokens and join them back into text
        
        # Apply the tokenization and store the first 8000 tokens in a new column 'limited_text'
        self.data['limited_text'] = self.data['text'].progress_apply(lambda x: get_first_tokens(x, max_tokens))

        # Filter entries where 'limited_text' has more than 10 tokens
        self.data['limited_text_length'] = self.data['limited_text'].apply(lambda x: len(x.split()))
        filtered_data = self.data[self.data['limited_text_length'] > 10]

        # Drop the temporary length column
        filtered_data.drop(columns=['limited_text_length'], inplace=True)

        return filtered_data

if __name__ == "__main__":
    file_path = os.path.join("./data", "euvsdisinfo_text.csv")
    processor = DataProcessor(file_path)
    processor.process_data()
    limited_data = processor.get_limited_token_data()
    output_path = os.path.join("./data", "cleaned_data.csv")
    limited_data.to_csv(output_path, index=False)


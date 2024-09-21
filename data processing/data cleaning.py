import pandas as pd
import re
import spacy

# Load the spaCy Russian model
nlp = spacy.load("ru_core_news_sm")

class DataProcessor:
    def __init__(self, file_path: str):
        self.data = pd.read_csv(file_path)
    
    def process_data(self):
        # Discard rows with missing values
        self.data.dropna(subset=["text"], inplace=True)
        
        # Apply cleaning to the "text" column
        self.data["text"] = self.data["text"].apply(self.clean_text)
    
    def clean_text(self, text):
        # Remove "Error retrieving content" messages
        text = re.sub(r"Error retrieving content: \d+ .+", "", text)
        
        # Remove date-time information (ISO 8601 format)
        text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\+\d{2}:\d{2}|Z)?", "", text)
        
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        
        # Remove text inside curly braces {{...}}
        text = re.sub(r"\{\{.*?\}\}", "", text)
        
        # Remove non-Cyrillic characters (keep punctuation and spaces)
        text = re.sub(r"[^\u0400-\u04FF\s.,!?;:]", "", text)
        
        # Remove extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def get_limited_token_data(self, max_tokens=7000):
        # Tokenize Russian text data using spaCy's ru_core_news_sm model
        self.data["token_count"] = self.data["text"].apply(lambda x: len([token.text for token in nlp(x)]))
        limited_data = self.data[self.data["token_count"] < max_tokens]
        return limited_data.drop(columns=["token_count"])

if __name__ == "__main__":
    processor = DataProcessor(r".\\data\\euvsdisinfo_text.csv")
    processor.process_data()
    limited_data = processor.get_limited_token_data()
    limited_data.to_csv("cleaned_data.csv", index=False)


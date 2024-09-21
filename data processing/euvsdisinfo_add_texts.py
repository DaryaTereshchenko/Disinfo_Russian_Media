import os
import requests
import pandas as pd
from bs4 import BeautifulSoup

class ExtractTextFromURL:
    def __init__(self, data:pd.DataFrame, lang: str):
        self.data = data
        self.lang = lang
    
    def get_subset_lang(self):
        try: 
            subset = self.data[self.data['article_language'] == self.lang]
            print(f"Subset of the data with the language: {self.lang}. Shape: {subset.shape}")
            return subset
        except Exception as e:
            return f"Error filtering data: {e}. Check the language code."
    
    def extract_text_from_url(self, url) -> str:
        try:
            response = requests.get(url)
            print(f"Processing the URL: {url}")
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(['script', 'style', 'img']):
                script.extract()
            text = soup.get_text(separator=' ')
            text = ' '.join(text.split())
            return text
        except Exception as e:
            return f"Error retrieving content: {e}"

    def add_text_column(self):
        subset = self.get_subset_lang()
        subset['text'] = subset['article_url'].apply(self.extract_text_from_url)
        return subset



if __name__ == '__main__':
    # Read the csv file with the raw data
    file_path_data = os.path.join('data', 'euvsdisinfo_base.csv')
    data = pd.read_csv(file_path_data)

    # Create a subset of the data with the specified language
    i = ExtractTextFromURL(data, 'Russian')
    result = i.add_text_column()

    # Save the result to a new CSV file
    file_path = os.path.join('data', 'euvsdisinfo_text.csv')
    result.to_csv(file_path, index=False, encoding='utf-8', errors='replace')

    print(f"Data saved to: {file_path}")
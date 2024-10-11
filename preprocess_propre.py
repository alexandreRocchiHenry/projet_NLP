from tqdm import tqdm
import numpy as np
import os
import glob
import nltk as nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import langid
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer



metadata_df = pd.read_csv('metadata.csv')


for i, filename in enumerate(glob.glob('data/txts/*.txt')):
    name = os.path.basename(filename).split('.')[0]
    with open(filename, 'r', encoding='utf-8') as file:
        file_content = file.read()
    
    index = metadata_df[metadata_df['doc_id'] == int(name)].index
    
    if not index.empty:
        metadata_df.at[index[0], 'text'] = file_content
    else:
        print(f"Le document ID '{name}' n'a pas été trouvé dans metadata_df")

print(metadata_df.head())

print(metadata_df['text'].isna().sum())
           
metadata_df.dropna(subset=['text'],inplace=True)

print(metadata_df['text'].isna().sum())

def get_language(text):
    if pd.isna(text) or len(text.strip()) == 0:
        return 'unknown'  
    langue, confiance = langid.classify(text)
    return langue
metadata_df['langue'] = metadata_df['text'].apply(get_language)

print(metadata_df.head())

print(metadata_df[metadata_df['langue']!='en'].head())

metadata_df = metadata_df[metadata_df['langue'] == 'en'] #supprimer les autres langues que l'anglais


tfidf_vectorizer = TfidfVectorizer()
#vectorizer = TfidfVectorizer(ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(metadata_df['text'])

metadata_df['tfidf'] = list(tfidf_matrix.toarray())

print(metadata_df['text'].head())


def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = re.sub(r'\d+\s*[\$€£]\s*\d+', '', text) 
    text = re.sub(r'©\s*\d{4}', '', text) 

    
    text = re.sub(r'[\n\t]+', ' ', text)   # Remplacer les retours 
    text = re.sub(r'\s+', ' ', text)  # Remplacer les espaces multiples


    tokens = word_tokenize(text)
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

metadata_df['text_processed'] = metadata_df['text'].apply(preprocess_text)
print('\npreprocessed\n')
print(metadata_df['text_processed'].head(5))


metadata_df.to_csv('data_preprocessed.csv', index=False)

from tqdm import tqdm
import numpy as np
import os
import glob
import nltk as nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import langid
import pandas as pd

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
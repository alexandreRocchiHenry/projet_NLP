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
from bs4 import BeautifulSoup

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


def is_css_document(text):
    """
    Vérifie si un document contient principalement du CSS en se basant sur la fréquence des éléments CSS.
    """
    # Indicateurs courants dans les blocs CSS
    css_indicators = ['{', '}', 'margin', 'padding', 'font-size', 'color', ':root', '@media']
    
    # Compter combien de fois ces éléments apparaissent dans le texte
    css_count = sum(text.count(indicator) for indicator in css_indicators)
    
    # Si le nombre d'éléments CSS dépasse un certain seuil par rapport à la longueur du texte, on considère que c'est du CSS
    if css_count > 5 and css_count / len(text) > 0.05:  # Ajustez le seuil selon vos données
        return True
    return False

# Appliquer le filtrage directement à la colonne 'text'
metadata_df = metadata_df[metadata_df['text'].apply(is_css_document) == False]


def preprocess_text(text):
    text = text.lower()

    text = re.sub(r'(#+\S+|@\S+|\s+@\S+|http\S+|word01|word02|[^A-Za-z0-9\'\’ ]+)', "", text)


    text = re.sub(r'[\n\t]+', ' ', text)
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Remplacer les espaces multiples


    tokens = word_tokenize(text)
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text
print('preprocessing')
metadata_df['text_processed'] = metadata_df['text'].apply(preprocess_text)
print('\npreprocessed\n')
print(metadata_df['text_processed'].head(5))
print('preprocessed, writning csv')

metadata_df.to_csv('data_preprocessed.csv', index=False)

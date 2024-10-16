import os
import glob
import nltk as nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import langid
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


metadata_df = pd.read_csv('metadata.csv')

list_file = sorted(glob.glob('data/txts/*.txt'), key=lambda x: int(os.path.basename(x).split('.')[0]))

for i, filename in enumerate(list_file):

    name = os.path.basename(filename).split('.')[0]
    index = metadata_df[metadata_df['doc_id'] == int(name)].index
    if not index.empty:
        with open(filename, 'r', encoding='utf-8') as file:
            file_content = file.read()
            metadata_df.at[index[0], 'text'] = file_content
    else:
        print(f"Le document ID '{name}' n'a pas été trouvé dans metadata_df")

metadata_df.dropna(subset=['text'],inplace=True)

def get_language(text):
    if pd.isna(text) or len(text.strip()) == 0:
        return 'unknown'  
    langue, confiance = langid.classify(text)
    return langue

metadata_df['langue'] = metadata_df['text'].apply(get_language)
metadata_df = metadata_df[metadata_df['langue'] == 'en']


def is_css_document(text):
    return not (re.search(r"{margin-|{display:",text) == None)

# Appliquer le filtrage directement à la colonne 'text'
metadata_df = metadata_df[metadata_df['text'].apply(is_css_document) == False]


def preprocess_text(text):
    #text = text.lower()
    #text = re.sub(r'(#+\S+|@\S+|\s+@\S+|http\S+|word01|word02|[^A-Za-z0-9\'\’ ]+)', "", text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    tokens = word_tokenize(text)
    cleaned_text = ' '.join(tokens)
    return cleaned_text

metadata_df['text_processed'] = metadata_df['text'].apply(preprocess_text)
print(metadata_df['text_processed'].head(5))

tfidf_vectorizer = TfidfVectorizer()
#ngram_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
tfidf_matrix = tfidf_vectorizer.fit_transform(metadata_df['text_processed'])
metadata_df['tfidf'] = list(tfidf_matrix.toarray())


#Preprocess sur les Institutions (= nos labels)

metadata_df["Institution"] = metadata_df['Institution'].replace('.',np.nan) 
metadata_df['Institution'] = metadata_df['Institution'].str.replace('.', '', regex=False).str.replace('?', '', regex=False)
metadata_df['Institution'] = metadata_df['Institution'].str.split(',').str[0].str.strip()
metadata_df['Institution'] = metadata_df['Institution'].str.split(';').str[0].str.strip()


metadata_df.to_csv('data_preprocessed.csv', index=False)



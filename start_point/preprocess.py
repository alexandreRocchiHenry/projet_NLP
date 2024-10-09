import glob
from tqdm import tqdm
import numpy as np
import os

import nltk as nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


corpus = []
stop_words = set(stopwords.words('english'))
OUT_FOLDER = "preprocessed/"

# Create output directory if it does not exist
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

print(f'Preprocessing...')
for i, filename in enumerate(tqdm(glob.glob('txts/*.txt'))):
    name = filename.split('/')[1].split('.')[0]
    with open(filename) as f:
        lines = f.read().strip()
        # Tokenize
        tokens = word_tokenize(lines)
        # Remove tokens with length < 3, not a link and not in stop words
        tokens = (' ').join([t.lower() for t in tokens
            if len(t) >= 3 
            and (t.isalpha() or t in "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~")
            and t.lower() not in stop_words 
            and not "http" in t.lower()
        ])

        # ngrams ?

        # Save tokens
        corpus.append(tokens)


# TF-IDF
def tfidf_filter(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    tfidf_values = np.array(X.mean(axis=0))[0]
    median_tfidf = np.quantile(tfidf_values, 0.5)
    mask = tfidf_values > median_tfidf
    words_to_keep = vectorizer.get_feature_names_out()[mask]
    print(type(words_to_keep))
    return words_to_keep

corpus_filt_tfidf = []
# words_to_keep = tfidf_filter(corpus)

for i, d in enumerate(tqdm(corpus)):
    words = d.split()
    # filt_words = [w for w in words if w in words_to_keep]
    # corpus_filt_tfidf.append(filt_words)
    f = open(f"{OUT_FOLDER}/{i}.txt", "w")
    f.write(" ".join(words) + "\n")
    f.close()

# with open('corpus_filt_tfidf.txt', 'w') as f:
#    for d in corpus_filt_tfidf:
#        doc = ' '.join(d)
#        f.write(doc + '\n')

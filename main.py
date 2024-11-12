import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import altair as alt

from nltk.tokenize import word_tokenize

from sklearn.decomposition import PCA

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering

import gensim.downloader as api
import nltk
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import torch
from transformers import AutoTokenizer, AutoModel,AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from transformers import RobertaModel, RobertaTokenizer


loaded_glove_model = api.load("glove-wiki-gigaword-300")
nltk.download('punkt')


# Embeddings functions
def tfidf(df):
    tfidf_vectorizer = TfidfVectorizer()
    Tfidf = tfidf_vectorizer.fit_transform(df['text_processed'])
    tfidf_a = Tfidf.toarray()
    return tfidf_a

def vocabulary_fct(corpus, voc_threshold):
    stopwords = nltk.corpus.stopwords.words('english')
    word_counts = {}
    for sent in corpus:
        for word in [word.lower() for word in word_tokenize(sent) if word.isalpha()]:
            if (word not in stopwords):
                if (word not in word_counts):
                    word_counts[word] = 0
                word_counts[word] += 1           
    words = sorted(word_counts.keys(), key=word_counts.get, reverse=True)
    if voc_threshold > 0:
        words = words[:voc_threshold] + ['UNK']   
    vocabulary = {words[i] : i for i in range(len(words))}
    return vocabulary, {word: word_counts.get(word, 0) for word in vocabulary}

def co_occurence_matrix(corpus, vocabulary, window=0, distance_weighting=False):
    stopwords = nltk.corpus.stopwords.words('english')
    l = len(vocabulary)
    cooc_matrix = np.zeros((l,l))
    for sent in corpus:
        # Get the sentence
        sent = [word.lower() for word in word_tokenize(sent) if word.isalpha()]
        # Obtain the indexes of the words in the sentence from the vocabulary 
        sent_idx = [vocabulary.get(word, len(vocabulary)-1) for word in sent if (word not in stopwords)]
        # Avoid one-word sentences - can create issues in normalization:
        if len(sent_idx) == 1:
                sent_idx.append(len(vocabulary)-1)
        # Go through the indexes and add 1 / dist(i,j) to M[i,j] if words of index i and j appear in the same window
        for i, idx in enumerate(sent_idx):
            # If we consider a limited context:
            if window > 0:
                # Create a list containing the indexes that are on the left of the current index 'idx_i'
                l_ctx_idx = [sent_idx[j] for j in range(max(0,i-window),i)]                
            # If the context is the entire document:
            else:
                # The list containing the left context is easier to create
                l_ctx_idx = sent_idx[:i]
            # Go through the list and update M[i,j]:        
            for j, ctx_idx in enumerate(l_ctx_idx):
                if distance_weighting:
                    weight = 1.0 / (len(l_ctx_idx) - j)
                else:
                    weight = 1.0
                cooc_matrix[idx, ctx_idx] += weight * 1.0
                cooc_matrix[ctx_idx, idx] += weight * 1.0
    return cooc_matrix

def pmi(co_oc, positive=True):
    sum_vec = co_oc.sum(axis=0)
    sum_tot = sum_vec.sum()
    with np.errstate(divide='ignore'):
        pmi = np.log((co_oc * sum_tot) / (np.outer(sum_vec, sum_vec)))                   
    pmi[np.isinf(pmi)] = 0.0  # log(0) = 0
    if positive:
        pmi[pmi < 0] = 0.0
    return pmi




def get_embedding(word, model):
    try:
        return model[word]
    except KeyError:
        return np.zeros(100)  

def glove_embeddings(df):
    all_embeddings = []
    for text in df['text_processed']:
        word_vectors = []
        for word in text.split():
            if word in loaded_glove_model:
                word_vectors.append(loaded_glove_model[word])
        if word_vectors:
            sentence_embedding = np.mean(word_vectors, axis=0)
        else:
            sentence_embedding = np.zeros(loaded_glove_model.vector_size)
        all_embeddings.append(sentence_embedding)
        all_embeddings_a = np.array(all_embeddings)
    return all_embeddings_a

def SVD_embeddings(df):
    svd = TruncatedSVD(n_components=300)
    texts = df['text'].values
    vocab,_  = vocabulary_fct(texts, 5000)
    M = co_occurence_matrix(texts, vocab, window=5, distance_weighting=False)
    SVDEmbeddings = svd.fit_transform(M)
    all_embeddings = []
    for texts in df['text_processed']:
        words = texts.split()
        word_indices = [vocab.get(word) for word in words if word in vocab]
        text_embeddings = [SVDEmbeddings[idx] for idx in word_indices if idx is not None]
        if text_embeddings:
            document_embedding = np.mean(text_embeddings, axis=0)
        else:
            document_embedding = np.zeros(svd.n_components)
        all_embeddings.append(document_embedding)
        all_embeddings_a = np.array(all_embeddings)
    return all_embeddings_a

def SVD_embeddings_PPMI(df):
    svd_ppmi = TruncatedSVD(n_components=300)
    texts = df['text'].values
    vocab,_  = vocabulary_fct(texts, 5000)
    M = co_occurence_matrix(texts, vocab, window=5, distance_weighting=False)
    PPMI = pmi(M)
    SVDEmbeddings = svd_ppmi.fit_transform(PPMI)
    all_embeddings = []
    for texts in df['text_processed']:
        words = texts.split()
        word_indices = [vocab.get(word) for word in words if word in vocab]
        text_embeddings = [SVDEmbeddings[idx] for idx in word_indices if idx is not None]
        if text_embeddings:
            document_embedding = np.mean(text_embeddings, axis=0)
        else:
            document_embedding = np.zeros(svd_ppmi.n_components)
        all_embeddings.append(document_embedding)
        all_embeddings_a = np.array(all_embeddings)
    return all_embeddings_a

def roberta_embeddings(df):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaModel.from_pretrained('roberta-large')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    embeddings = []
    for text in df['text_processed']:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    embeddings_array = np.array(embeddings)
    return embeddings_array

def sentence_transformer_embeddings(df):
    model_name='roberta-base-nli-stsb-mean-tokens'
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df['text_processed'].tolist(), show_progress_bar=True)
    return embeddings


# Dimension reduction functions
def tsne(embeddings):
    docs_tsne = TSNE(n_components=2, learning_rate='auto',
                init='pca').fit_transform(embeddings)
    return docs_tsne

def pca(embeddings):
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    return embeddings_pca

# Clusterings functions
def Kmeans_clustering(n_clusters, embeddings, model_name):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels

def gaussian_clustering(n_clusters, embeddings, model_name):
    if model_name == 'roberta_embeddings':
        n_clusters = n_clusters - 3
    if model_name == 'sentence_transformer_embeddings':
        n_clusters = n_clusters
    gmm = GaussianMixture(n_components=n_clusters, random_state=0, covariance_type='diag', reg_covar=1e-6)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)
    return labels

def hierarchical_clustering(n_clusters, embeddings, model_name):
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = hc.fit_predict(embeddings)
    return labels

def correspondence_analysis(embeddings, n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    embeddings_ca = svd.fit_transform(embeddings)
    return embeddings_ca

# Validation function
def score_function(embeddings, labels):
    silhouette_s = silhouette_score(embeddings, labels)
    davies_bouldin_s = davies_bouldin_score(embeddings, labels)
    calinski_harabasz_s = calinski_harabasz_score(embeddings, labels)
    return silhouette_s, davies_bouldin_s, calinski_harabasz_s

def display_ca(embeddings, df, labels, embedding_name, clustering_name):
    embeddings_ca = correspondence_analysis(embeddings)
    data_ca = pd.DataFrame({'x': embeddings_ca[:, 0],
                            'y': embeddings_ca[:, 1],
                            'institution': df['categorie Institution'],
                            'title': df["Name of the document"],
                            'labels': labels
                            })
    alt.data_transformers.disable_max_rows()
    chart = alt.Chart(data_ca).mark_circle(size=200).encode(
        x="x", y="y", color=alt.Color('labels:N', scale=alt.Scale(scheme='category20')),
        tooltip=['institution', "title"]
        ).interactive().properties(
        width=500,
        height=500
    )
    chart.save(f'T3_initial_clustering_{embedding_name}_{clustering_name}_CA.html')
    chart.show()
    

def display_pca(embeddings, df, labels, embedding_name, clustering_name):
    embeddings_pca = pca(embeddings)
    data_pca = pd.DataFrame({'x': embeddings_pca[:, 0],
                            'y': embeddings_pca[:, 1],
                            'institution': df['categorie Institution'],
                            'title': df["Name of the document"],
                            'labels': labels
                            })
    alt.data_transformers.disable_max_rows()
    chart = alt.Chart(data_pca).mark_circle(size=200).encode(
        x="x", y="y", color=alt.Color('labels:N', scale=alt.Scale(scheme='category20')),
        tooltip=['institution', "title"]
        ).interactive().properties(
        width=500,
        height=500
    )
    chart.save(f'T3_initial_clustering_{embedding_name}_{clustering_name}_PCA.html')
    chart.show()   

def display_tsne(embeddings, df, labels, embedding_name, clustering_name):
    docs_tsne_th = TSNE(n_components=2, learning_rate='auto',
                        init='random', metric='cosine',
                        perplexity=50.0).fit_transform(embeddings)
    print(docs_tsne_th.shape)

    data_th = pd.DataFrame({'x': docs_tsne_th[:,0],
                            'y': docs_tsne_th[:,1],
                            'institution': df['categorie Institution'],
                            'title': df["Name of the document"],
                            'labels': labels
                            #'labels': df["categorie Institution"]
                            #'labels': df["theme"]
                            })
    alt.data_transformers.disable_max_rows()
    chart = alt.Chart(data_th[:]).mark_circle(size=200).encode(
        x="x", y="y", color=alt.Color('labels:N', 
                                    scale=alt.Scale(scheme='category20')),
        tooltip=['institution', "title"]
        ).interactive().properties(
        width=500,
        height=500
    )
    chart.save(f'T3_initial_clustering_{embedding_name}_{clustering_name}_TSNE.html')
    chart.show()


# Clustering pipeline
def pipeline(dataframe, embedding_method, clustering_method, taille_cluster=[10,11], reduction_method=display_pca):
    print(f"start embedding for {embedding_method.__name__} and {clustering_method.__name__}")
    embeddings = embedding_method(dataframe)
    print("clustering")
    for i in range(taille_cluster[0], taille_cluster[1]):
        labels = clustering_method(i, embeddings, embedding_method.__name__)
    print("scoring")
    scores = score_function(embeddings, labels)
    print(f"silhouette_score: {scores[0]}, davies_bouldin_score: {scores[1]}, calinski_harabasz_score: {scores[2]}")
    reduction_method(embeddings, dataframe, labels, embedding_method.__name__, clustering_method.__name__)
    return scores



def main():
    # Preprocessed dataframe
    data_proprocessed = "Data_csv/data_preprocessed.csv"
    data_df = pd.read_csv(data_proprocessed)

    Clustering_methods = [Kmeans_clustering, hierarchical_clustering]
    Embedding_methods = [glove_embeddings, SVD_embeddings, SVD_embeddings_PPMI, sentence_transformer_embeddings]
    reduction_methods = [display_pca, display_tsne, display_ca]

    results = []

    for embedding_method in Embedding_methods:
        for cluster_method in Clustering_methods:
            for reduction in reduction_methods:
                
                result = pipeline(dataframe=data_df, 
                                embedding_method=embedding_method,
                                clustering_method=cluster_method,
                                reduction_method=reduction
                                )
                
                results.append({
                    'Embedding Method': embedding_method.__name__,
                    'Clustering Method': cluster_method.__name__,
                    'silhoutte score': result[0],  
                    'davies score' : result[1], 
                    'calinski score' : result[2], 
                })

        # Conversion des résultats en DataFrame
        results_df = pd.DataFrame(results)

        # Sauvegarde des résultats dans un fichier CSV
        results_df.to_csv(f'pipeline_results_{reduction}.csv', index=False)



    return

if __name__ == "__main__":
    main()
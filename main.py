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

from sklearn.decomposition import PCA

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering



# Preprocessed dataframe
data_proprocessed = "data_preprocessed.csv"
data_df = pd.read_csv(data_proprocessed)

# Embeddings functions
def tfidf(df):
    tfidf_vectorizer = TfidfVectorizer()
    Tfidf = tfidf_vectorizer.fit_transform(df['text_processed'])
    tfidf_a = Tfidf.toarray()
    return tfidf_a

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
def Kmeans_fct(n, embeddings):
    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    return labels


def correspondence_analysis(embeddings, n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    embeddings_ca = svd.fit_transform(embeddings)
    return embeddings_ca

def hierarchical_clustering(n_clusters, embeddings):
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    labels = hc.fit_predict(embeddings)
    return labels

# Validation function
def score_function(embeddings, labels):
    silhouette_s = silhouette_score(embeddings, labels)
    davies_bouldin_s = davies_bouldin_score(embeddings, labels)
    calinski_harabasz_s = calinski_harabasz_score(embeddings, labels)
    return silhouette_s, davies_bouldin_s, calinski_harabasz_s

def display_ca(embeddings, df, labels):
    embeddings_ca = correspondence_analysis(embeddings)
    data_ca = pd.DataFrame({'x': embeddings_ca[:, 0],
                            'y': embeddings_ca[:, 1],
                            'institution': df['categorie Institution'],
                            'title': df["Name of the document"],
                            'labels': df["categorie Institution"]
                            })
    alt.data_transformers.disable_max_rows()
    chart = alt.Chart(data_ca).mark_circle(size=200).encode(
        x="x", y="y", color=alt.Color('labels:N', scale=alt.Scale(scheme='category20')),
        tooltip=['institution', "title"]
        ).interactive().properties(
        width=500,
        height=500
    )
    chart.save('chart.html')
    chart.show()
    

def display_pca(embeddings, df, labels):
    embeddings_pca = pca(embeddings)
    data_pca = pd.DataFrame({'x': embeddings_pca[:, 0],
                            'y': embeddings_pca[:, 1],
                            'institution': df['categorie Institution'],
                            'title': df["Name of the document"],
                            'labels': df["categorie Institution"]
                            })
    alt.data_transformers.disable_max_rows()
    chart = alt.Chart(data_pca).mark_circle(size=200).encode(
        x="x", y="y", color=alt.Color('labels:N', scale=alt.Scale(scheme='category20')),
        tooltip=['institution', "title"]
        ).interactive().properties(
        width=500,
        height=500
    )
    chart.save('chart.html')
    chart.show()   

def display_tsne(embeddings, df, labels):
    docs_tsne_th = TSNE(n_components=2, learning_rate='auto',
                        init='random', metric='cosine',
                        perplexity=50.0).fit_transform(embeddings)
    print(docs_tsne_th.shape)

    data_th = pd.DataFrame({'x': docs_tsne_th[:,0],
                            'y': docs_tsne_th[:,1],
                            'institution': df['categorie Institution'],
                            'title': df["Name of the document"],
                            'labels': df["categorie Institution"]
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
    chart.save('chart.html')
    chart.show()


# Clustering pipeline
def pipeline(dataframe, embedding_method, clustering_method, taille_cluster, reduction_method=display_tsne):
    print("start embedding")
    embeddings = embedding_method(dataframe)
    print("clustering")
    for i in range(taille_cluster[0], taille_cluster[1]):
        labels = clustering_method(i, embeddings)
    print("scoring")
    scores = score_function(embeddings, labels)
    print(f"silhouette_score: {scores[0]}, davies_bouldin_score: {scores[1]}, calinski_harabasz_score: {scores[2]}")
    reduction_method(embeddings, dataframe, labels)
    return scores

pipeline(dataframe=data_df, embedding_method=tfidf, clustering_method=Kmeans_fct, taille_cluster=[10,11], reduction_method=display_tsne)

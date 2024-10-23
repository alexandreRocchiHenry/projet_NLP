import re
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import altair as alt

data_proprocessed = "data_preprocessed.csv"
data_df = pd.read_csv(data_proprocessed)

def tfidf(df):
    tfidf_vectorizer = TfidfVectorizer()
    Tfidf = tfidf_vectorizer.fit_transform(df['text_processed'])
    tfidf_a = Tfidf.toarray()
    return tfidf_a




def score_function(embeddings, labels):
    silhouette_s = silhouette_score(embeddings, labels)
    davies_bouldin_s = davies_bouldin_score(embeddings, labels)
    calinski_harabasz_s = calinski_harabasz_score(embeddings, labels)
    return silhouette_s, davies_bouldin_s, calinski_harabasz_s

def display_ca(embeddings, df, labels):
    data_ca = pd.DataFrame({'x': embeddings[:, 0],
                            'y': embeddings[:, 1],
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
    chart.save('chart.html')
    chart.show()

def pipeline(dataframe, embedding_method, clustering_method, taille_cluster):
    print("start embedding")
    embeddings = embedding_method(dataframe)
    embeddings_ca = correspondence_analysis(embeddings)
    print("clustering")
    for i in range(taille_cluster[0], taille_cluster[1]):
        labels = clustering_method(i, embeddings_ca)
    print("scoring")
    scores = score_function(embeddings_ca, labels)
    print(f"silhouette_score: {scores[0]}, davies_bouldin_score: {scores[1]}, calinski_harabasz_score: {scores[2]}")
    display_ca(embeddings_ca, dataframe, labels)
    return scores

pipeline(dataframe=data_df, embedding_method=tfidf, clustering_method=hierarchical_clustering, taille_cluster=[6,7])

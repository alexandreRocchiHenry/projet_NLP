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
import prince

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

def create_contingency_table(df, cluster_labels, theme_labels):
    df_temp = df.copy()
    df_temp['Cluster'] = cluster_labels
    df_temp['Theme'] = theme_labels
    contingency_table = pd.crosstab(df_temp['Cluster'], df_temp['Theme'])
    return contingency_table


def perform_correspondence_analysis(contingency_table):
    ca = prince.CA(n_components=2, n_iter=10, copy=True, check_input=True, engine='auto')
    ca = ca.fit(contingency_table)
    row_coords = ca.row_coordinates(contingency_table)
    col_coords = ca.column_coordinates(contingency_table)
    return ca, row_coords, col_coords

def plot_ca_results(row_coords, col_coords, embedding_name, clustering_name):
    fig, ax = plt.subplots(figsize=(10, 8))

   
    ax.scatter(row_coords[0], row_coords[1], c='blue', label='Clusters')
    for i, txt in enumerate(row_coords.index):
        ax.annotate(txt, (row_coords.iloc[i, 0], row_coords.iloc[i, 1]))


    ax.scatter(col_coords[0], col_coords[1], c='red', label='Themes', marker='x')
    for i, txt in enumerate(col_coords.index):
        ax.annotate(txt, (col_coords.iloc[i, 0], col_coords.iloc[i, 1]))

    ax.legend()
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Correspondence Analysis')
    plt.savefig(f'T4_initial_clustering_{embedding_name}_{clustering_name}_CA.png')
    plt.show()


def display_ca(embeddings, cluster_labels, df, embedding_name, clustering_name):
    theme_labels = df['theme']
    contingency_table = create_contingency_table(df, cluster_labels, theme_labels)
    ca, row_coords, col_coords = perform_correspondence_analysis(contingency_table)
    plot_ca_results(row_coords, col_coords, embedding_name, clustering_name)


def pca(embeddings):
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_ 
    print(f"Explained variance: {explained_variance}")
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

# def display_ca(embeddings, df, labels, embedding_name, clustering_name):
#     embeddings_ca = correspondence_analysis(embeddings)
#     data_ca = pd.DataFrame({'x': embeddings_ca[:, 0],
#                             'y': embeddings_ca[:, 1],
#                             'institution': df[labels],
#                             'title': df["Name of the document"],
#                             'labels': df[labels]
#                             })
#     alt.data_transformers.disable_max_rows()
#     chart = alt.Chart(data_ca).mark_circle(size=200).encode(
#         x="x", y="y", color=alt.Color('labels:N', scale=alt.Scale(scheme='category20')),
#         tooltip=['institution', "title"]
#         ).interactive().properties(
#         width=500,
#         height=500
#     )
#     chart.save(f'T4_initial_clustering_{embedding_name}_{clustering_name}_CA.html')
#     chart.show()
    

def display_pca(embeddings, df, labels, embedding_name, clustering_name):
    embeddings_pca = pca(embeddings)
    data_pca = pd.DataFrame({'x': embeddings_pca[:, 0],
                            'y': embeddings_pca[:, 1],
                            'institution': df["Institution"],
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
    chart.save(f'T5_initial_clustering_{embedding_name}_{clustering_name}_PCA.html')
    chart.show()   

def display_pca_with_topics(embeddings, df, embedding_name, clustering_name, topic_keywords):
    embeddings_pca = pca(embeddings)
    data_pca = pd.DataFrame({
        'x': embeddings_pca[:, 0],
        'y': embeddings_pca[:, 1],
        'Cluster': df['Cluster'],
        'Topic': df['Topic'],
        'Text': df['text_processed']
    })
    
    # Créer une colonne combinée pour Cluster et Topic pour une meilleure distinction
    data_pca['Cluster_Topic'] = data_pca['Cluster'].astype(str) + '_' + data_pca['Topic'].astype(str)
    
    # Préparer les mots-clés des topics pour les annotations
    topic_labels = {}
    for cluster_label in topic_keywords:
        topics = topic_keywords[cluster_label]
        for idx, keywords in enumerate(topics):
            label = f"Cluster {cluster_label}, Topic {idx}"
            words = ', '.join(keywords)
            topic_labels[f"{cluster_label}_{idx}"] = words
    
    # Ajouter les mots-clés en tant que labels
    data_pca['Topic_Keywords'] = data_pca['Cluster_Topic'].map(topic_labels)
    
    # Visualisation avec Altair
    alt.data_transformers.disable_max_rows()
    chart = alt.Chart(data_pca).mark_circle(size=60).encode(
        x='x',
        y='y',
        color=alt.Color('Cluster_Topic:N', scale=alt.Scale(scheme='category20')),
        tooltip=['Text', 'Topic_Keywords']
    ).properties(
        width=800,
        height=600,
        title=f'PCA Visualization with LDA Topics ({embedding_name}, {clustering_name})'
    ).interactive()
    
    chart.save(f'PCA_with_LDA_Topics_{embedding_name}_{clustering_name}.html')
    chart.show()



def display_tsne(embeddings, df, labels, embedding_name, clustering_name):
    docs_tsne_th = TSNE(n_components=2, learning_rate='auto',
                        init='random', metric='cosine',
                        perplexity=50.0).fit_transform(embeddings)
    print(docs_tsne_th.shape)

    data_th = pd.DataFrame({'x': docs_tsne_th[:,0],
                            'y': docs_tsne_th[:,1],
                            'institution': df[labels],
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
    chart.save(f'T5_initial_clustering_{embedding_name}_{clustering_name}_TSNE.html')
    chart.show()


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def extract_topic_keywords(lda_model, vectorizer, n_top_words=10):
    keywords = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        keywords.append(top_features)
    return keywords

def lda_on_clusters(df, n_topics=5):
    cluster_labels = df['Cluster'].unique()
    lda_models = {}
    vectorizers = {}
    topic_keywords = {}
    
    for label in cluster_labels:
        cluster_data = df[df['Cluster'] == label]
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(cluster_data['text_processed'])
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda.fit(dtm)
        
        lda_models[label] = lda
        vectorizers[label] = vectorizer
        
        # Attribution des topics aux documents du cluster
        topic_values = lda.transform(dtm)
        df.loc[df['Cluster'] == label, 'Topic'] = topic_values.argmax(axis=1)
        
        # Extraction des mots-clés pour chaque topic
        keywords = extract_topic_keywords(lda, vectorizer)
        topic_keywords[label] = keywords
    
    return df, lda_models, vectorizers, topic_keywords




# Clustering pipeline
def pipeline(dataframe, embedding_method, clustering_method, reduction_method, taille_cluster=[10,11]):
    print(f"Start embedding with {embedding_method.__name__}")
    embeddings = embedding_method(dataframe)
    print("Clustering")
    for i in range(taille_cluster[0], taille_cluster[1]):
        labels = clustering_method(i, embeddings, embedding_method.__name__)
    dataframe['Cluster'] = labels
    print("Scoring")
    scores = score_function(embeddings, labels)
    print(f"Silhouette Score: {scores[0]}, Davies-Bouldin Score: {scores[1]}, Calinski-Harabasz Score: {scores[2]}")
    
    print("Applying LDA on Clusters")
    dataframe, lda_models, vectorizers, topic_keywords = lda_on_clusters(dataframe)
    
    print("Displaying PCA with LDA Topics")
    display_pca_with_topics(embeddings, dataframe, embedding_method.__name__, clustering_method.__name__, topic_keywords)
    
    return scores



def main():
    # Preprocessed dataframe
    data_proprocessed = "Data_csv/data_preprocessed.csv"
    data_df = pd.read_csv(data_proprocessed)

    Clustering_methods = [Kmeans_clustering]
    Embedding_methods = [sentence_transformer_embeddings]
    reduction_methods = [display_pca]

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
                    'Reduction Method': reduction.__name__,
                    'silhoutte score': result[0],  
                    'davies score' : result[1], 
                    'calinski score' : result[2],
                     
                })

        # Conversion des résultats en DataFrame
        results_df = pd.DataFrame(results)

        # Sauvegarde des résultats dans un fichier CSV
        results_df.to_csv(f'pipeline_results_{reduction}_2.csv', index=False)



    return

if __name__ == "__main__":
    main()
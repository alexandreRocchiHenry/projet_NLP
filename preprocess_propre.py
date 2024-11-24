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
import ujson as json
from tqdm import tqdm
import string

tqdm.pandas()
nltk.download('stopwords')
nltk.download('punkt')


metadata_df = pd.read_csv('Data_csv/metadata.csv')

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

from bs4 import BeautifulSoup

def is_html_document(text):
    soup = BeautifulSoup(text, "html.parser")
    # Si le contenu textuel est très faible par rapport au contenu total, c'est probablement du code HTML/CSS
    text_length = len(soup.get_text(strip=True))
    total_length = len(text)
    return text_length / total_length < 0.5  # Seuil à ajuster

def is_wp_preset_document(text):
    # Rechercher les occurrences de '-- wp -- preset --'
    preset_pattern = r'--\s*wp\s*--\s*preset\s*--'
    matches = re.findall(preset_pattern, text, re.IGNORECASE)
    return len(matches) > 5 

def is_css_document(text):
    css_pattern = r'[^\n]*\{\s*[^}]*\s*\}'
    matches = re.findall(css_pattern, text)
    return len(matches) > 5

def is_code_like_text(text):
    total_chars = len(text)
    alnum_chars = sum(c.isalnum() or c.isspace() for c in text)
    non_alnum_ratio = (total_chars - alnum_chars) / total_chars
    return non_alnum_ratio > 0.3

def is_meaningful_text(text):
    if pd.isna(text) or len(text.strip()) == 0:
        return False
    language, confidence = langid.classify(text)
    confidence = confidence if language == 'en' else 0
    confidence = abs(confidence)
    return confidence > 100

def is_valid_document(text):
    if not is_meaningful_text(text):
        return False
    if is_css_document(text):
        return False
    if is_code_like_text(text):
        return False
    if is_html_document(text):
        return False
    if is_wp_preset_document(text):
        return False
    return True

# Apply the composite validation function
metadata_df = metadata_df[metadata_df['text'].apply(is_valid_document)]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation] 
    
    stopwords = nltk.corpus.stopwords.words('english')
    
    tokens = [word for word in tokens if word not in stopwords]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

metadata_df['text_processed'] = metadata_df['text'].progress_apply(preprocess_text)
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

metadata_df = metadata_df.dropna(subset=['Institution'])

categories = {
    "Gouvernements et organismes publics nationaux": [
        "Government", "Ministry", "Department", "Authority", "Council", "Office", "Agency", 
        "Republic", "State", "Presidency", "Embassy", "White House", "Agenzia per l'Italia Digitale",
        "Autorité de contrôle prudentiel et de résolution", "Canada", "TBS Canada", "Prime Minister of Malaysia",
        "Ministerio de Asuntos Economicos y Transformacion Digital", "Executive Yuan", "Poland", "India"
    ],
    "Organisations internationales et institutions supranationales": [
        "United Nations", "European Union", "Council of Europe", "OECD", "G7", "G20", 
        "Commission", "Parliament", "World Bank", "Organization", "NATO", "UNESCO",
        "Think 20", "The Greens", "Inter-American Development Bank", "Organisation",
        "Organization", "Eurocontrol","CIGI"
    ],
    "Entreprises technologiques et multinationales": [
        "Google", "Intel", "Microsoft", "IBM", "Salesforce", "Apple", "Amazon", "Facebook", 
        "Thales", "Axon", "Philips", "BlackBerry", "Aptiv", "Thomson Reuters", "Tieto", "BCG",
        "Telia Company", "Bitkom", "Smart Dubai", "QuantumBlack", "PriceWaterhouseCoopers", "Deutsche Telekom",
        "KPMG", "Mc Kinsey", "Orange", "Dataiku", "IntegrateAI", "Ipsos", "IDEO", "Indie","Examai","fastai"
    ],
    "Instituts de recherche et universités": [
        "Institute", "University", "Academy", "MIT", "Stanford", "Harvard", "Oxford", 
        "Cambridge", "Fraunhofer", "Future of Humanity Institute", "Research Center", 
        "AI Institute", "School", "Adapt Center", "Université de Montréal", 
        "TU Wien", "Vrije Universiteit Brussel", "American College of Radiology", "The Rathenau Instituut",
        "Centre for the Governance of AI", "CERNA",
        "CIFAR", "ETH Zurich", "KU Leuven", "Markkula Center", "National Academies of Science", "Plos Computational Biology","Laboratoire"
    ],
    "Comités éthiques et régulateurs": [
        "Ethics", "Council", "Commission", "Supervisory", "Advisory", "Regulator", 
        "Data Protection", "Privacy Commission", "Rights Commission", "Ethikkommission",
        "AI Recht", "Z-Inspection", "AlgorithmWatch", "Article 29 Working Party", "Défenseur des Droits",
        "ESMA", "European Digital Rights", "European Network of Equality Bodies"
    ],
    "Associations professionnelles et groupes de réflexion": [
        "Association", "Forum", "Think Tank", "ACM", "IEEE", "Society", "Tech Alliance", 
        "Tech Group", "Advocacy Group", "Council", "AI4People", "AISP", "Women Leading in AI",
        "All Tech is Human", "ALLAI", "UX Studio Team", "Always Designing for People", "ARMAI",
        "Bertelsmann Stiftung", "The European Robotics Research Network", "The Information Accountability Foundation",
        "Brookings", "Reghorizon", "Renaissance Foundation", "Center for Democracy & Technology", "Center for Humane Technology",
        "CIGREF/Syntec Numérique", "Konrad-Adenauer-Stiftung", "Hub4NGI", "Open Loop", "O'Reilly", "Mozilla Foundation", "Numeum"
    ],
    "ONG et initiatives de droits humains": [
        "Human Rights", "Privacy", "Access Now", "accessnow", "Article 19", "Amnesty International", 
        "Open Rights Group", "Freedom Online Coalition", "Humanitarian", "Advocacy", "The Public Voice",
        "Federation of German Consumer Organisations"
    ],
    "Organisations intersectorielles et partenariats publics-privés": [
        "Partnership", "Alliance", "Initiative", "Coalition", "Collaboration", "Consortium", 
        "Task Force", "Forum", "Global", "Joint", "Group", "WEF", "W20"
    ],
    "Organismes normatifs et de standardisation": [
        "ISO", "Standards", "Standardization", "Regulation", "European Law Institute", 
        "Certification", "Norms", "Accreditation", "Compliance"
    ],
    "Conseils et groupes consultatifs gouvernementaux": [
        "AI Council", "Board", "Advisory", "Task Force", "Council", "Mission", 
        "Strategy Group", "Committee", "AI Strategy", "Digital Service", "IAPP"
    ]
}


def categorize_organization(org):
    org = org.strip().lower()
    for category, keywords in categories.items():

        if any(keyword.lower() in org for keyword in keywords):
            return category
    return "Autres"

metadata_df['categorie Institution'] = metadata_df['Institution'].apply(categorize_organization)

# Charger le fichier JSON contenant les mots-clés des thèmes
with open('start_point/themes.json', 'r') as f:
    keywords = json.load(f)

feature_names = tfidf_vectorizer.get_feature_names_out()

# Fonction pour attribuer le thème basé sur le plus grand score TF-IDF des mots-clés
def assign_theme_with_highest_tfidf(text, keywords):
    # Transformer le texte en un vecteur TF-IDF
    tfidf_vector = tfidf_vectorizer.transform([text])
    tfidf_scores = tfidf_vector.toarray()[0]
    
    theme_tfidf_scores = {}

    for theme, kw_list in keywords.items():
        # Trouver les scores TF-IDF pour les mots-clés de chaque thème
        theme_scores = [tfidf_scores[feature_names.tolist().index(kw.lower())] 
                        for kw in kw_list if kw.lower() in feature_names]
        
        # Prendre le plus grand score TF-IDF de ce thème
        if theme_scores:
            theme_tfidf_scores[theme] = max(theme_scores)

    # Retourner le thème ayant le score TF-IDF le plus élevé
    if theme_tfidf_scores:
        max_theme = max(theme_tfidf_scores, key=theme_tfidf_scores.get)
        return max_theme
    else:
        return 'Aucun thème'

# 3. Ajouter une nouvelle colonne 'theme' avec le thème ayant le mot-clé au score TF-IDF le plus élevé
metadata_df['theme'] = metadata_df['text_processed'].progress_apply(lambda text: assign_theme_with_highest_tfidf(text, keywords))

metadata_df.to_csv('Data_csv/data_preprocessed.csv', index=False)



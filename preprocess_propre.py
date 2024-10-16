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






# Fonction pour trouver la catégorie d'une organisation en fonction des mots-clés
def categorize_organization(org):
    org = org.strip().lower()
    for category, keywords in categories.items():

        if any(keyword.lower() in org for keyword in keywords):
            return category
    return "Autres"

# Appliquer la fonction sur la colonne 'organization' et créer une nouvelle colonne 'category'
metadata_df['categorie Institution'] = metadata_df['Institution'].apply(categorize_organization)




metadata_df.to_csv('data_preprocessed.csv', index=False)



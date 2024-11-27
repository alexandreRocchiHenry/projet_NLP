# Projet Natural Language Processing (NLP) - clusterisation manifestos/chartes IA Ethiques

#### par Alexandre Malfoy, Damien Thai, Baptiste Cervoni, Alexandre Rocchi

Avec le développement de l’intelligence artificielle (IA), une prise conscience générale de ses enjeux et de ses problématiques a vu le jour. En effet, de nombreuses institutions et organismes de divers secteurs se sont intéressés à la question de l’éthique et la réglementation dans l’IA. 
Durant ce projet, la collection MapAIE de 627 chartes et manifestes autour de l'intelligence artificielle et de l'éthique de l'IA sera étudiée afin d’identifier les principaux thèmes abordés de regrouper les articles selon ceux-ci. Le travail effectué dans ce rapport se basera sur l'article [GORNET Mélanie, DELARUE Simon, BORITCHEV Maria, VIARD Tiphaine, Mapping AI ethics: a meso-scale analysis of its charters and manifestos](/mapaie-paper.pdf), The 2024 ACM Conference on Fairness, Accountability, and Transparency. 2024. p. 127-140 qui traite de la construction du corpus étudié et fournissant une première analyse des thèmes traités. 
Le projet se décompose en quatres grands axes majeurs, le prétraitement des données, la vectorisation des documents, la clusterisation et topic modeling ainsi que la visualisation de ces clusters. Pour chacune des étapes, plusieurs techniques de ont été utilisées afin d’établir une pipeline optimale pour la clusterisation des articles du corpus.


### Instalation et Configuration


1- cloner le git


    git clone git@github.com:alexandreRocchiHenry/Animals-Cassification.git


2- télécharger les dépendances avec pip

    pip install -r requirements.txt

3- dans le dossier pipeline_startpoint, lancer le script [pipeline_start_point.py](/pipeline_startpoint/pipeline_start_point.py)
    
    cd pipeline_startpoint
    python3 pipeline_start_point.py

4- dans le dossier principal lancer [preprocess_propre.py](/preprocess_propre.py)
  
    cd ..
    python3 preprocess_propre.py

- utiliser le [main.py](/main.py) contenant le code pour lancer le clustering. Ce code peut être modifié pour prendre différentes méthodes 
d'embedings, de clustering et réduction de dimension. 

- differents notebook permettent aussi de lancer [BERTopic](/BERTopic.ipynb), [GPTopic](/GPTopic.ipynb), [LDA](/LDA.ipynb) et [detailled_analysis](/detailled_analysis.ipynb)
  



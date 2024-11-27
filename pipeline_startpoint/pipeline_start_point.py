import requests
import os
from random import choice
from tqdm import tqdm
import csv
import glob
import magic
from Parser import Parser
import argparse
import numpy
import ujson as json

requests.packages.urllib3.disable_warnings()

URL_FILE = "list_urls.txt"
MANIFESTOS_FILE = "all_manifestos.csv"
UA_FILE = "user_agents.txt"
OUT_FOLDER = "../data/docs"
LOG_FILE = "dl_docs.log"
log_fp = open(LOG_FILE, "w", encoding="utf8")


def csv_to_dict(filepath):
    manifestos = {}
    with open(filepath, encoding="utf8") as f:
        data = csv.reader(f)
        headers = next(data)
        manifestos_list = []

        for d in data:
            manifesto = { h: "" for h in headers }
            for i,x in enumerate(d):
                head = headers[i]
                manifesto[head] = x
            manifestos_list.append(manifesto)
    return manifestos_list

manifestos_list = csv_to_dict(MANIFESTOS_FILE)
list_of_urls = [ x["URL"] for x in manifestos_list] #if x["Status"].lower() == "included" ]
user_agents = [ x.strip() for x in open(UA_FILE,  encoding="utf8").readlines() ] 

# Create output directory if it does not exist
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

f_metadata = open("mapaie-metadata.csv", "w", encoding="utf8")

for i in tqdm(range(len(manifestos_list))):
    manifesto = manifestos_list[i]
    title = manifesto["Name of the document"]
    institution = manifesto["Institution"]
    url = manifesto["URL"]

    try:
        headers = { "User-Agent": choice(user_agents), "Referer": "http://perdu.com" }
        response = requests.get(url, headers=headers, timeout=10, verify=False)
    except requests.exceptions.RequestException as e:
        print(f"ERR: {url}, {e}", file=log_fp)

    if response.status_code == 200:
        print(f"{url},OK", file=log_fp)
        if url[-4:] == ".pdf":
            with open(f"{OUT_FOLDER}/{i}.pdf", "wb") as f:
                f.write(response.content)
        else:
            with open(f"{OUT_FOLDER}/{i}.html", "wb") as f:
                f.write(response.content)
        f_metadata.write(f"{i}|{title}|{institution}\n")

    else:
        # if we received any error http code
        print(f"ERR: {url},{response.status_code}", file=log_fp)

log_fp.close()
f_metadata.close()


LOG_FILE = "parse.log"
OUT_FOLDER = "../data/txts"
log_fp = open(LOG_FILE, "w")

p = Parser(log_file=log_fp)

# Create output directory if it does not exist
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

all_files = [f for f in glob.glob("../data/docs/*")]

for i in tqdm(range(len(all_files))):
    fname = all_files[i]
    ftype = magic.from_file(fname, mime=True)

    if ftype == "text/html" or ftype == "text/xml":
        # this is a html file
        p.parse_html(fname)
    elif ftype == "application/pdf":
        # this is a pdf file
        p.parse_pdf(fname)
    else:
        print(f"ERR. NOT A RECOGNIZED FILETYPE: {fname}, {ftype}.", file=log_fp)

log_fp.close()


LOG_FILE = "corpus.log"
OUT_FILE = "corpus.txt"
log_fp = open(LOG_FILE, "w", encoding="utf-8")



# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data")
parser.add_argument("-m", "--method", choices=["iramuteq", "cortext"], default = "iramuteq")
parser.add_argument("-t", "--themes", default="themes.json")
# parser.add_argument("-d", "--destination", choices=["iramuteq", "cortext"])

args = parser.parse_args()
print(args)

corpus_file = open(OUT_FILE, "w", encoding="utf-8")
nb_docs = 0

# Keywords
keywords = json.load(open(args.themes, encoding="utf-8"))

filt_crit = lambda x, kw_list: all(x)

iramuteq = False
cortext = False


if args.method == "iramuteq":
    iramuteq = True
    cortext = False
elif args.method == "cortext":
    cortext = True
    iramuteq = False

    for t in keywords:
        if not os.path.exists(t):
            os.makedirs(t)
else:
    iramuteq = True

# Counting docs per theme
doc_counts = { k: 0 for k, v in keywords.items() }
doc_occurrences = {}

for i, fname in enumerate(tqdm(glob.glob(f"./{args.data}/*.txt"))):
    nb_docs += 1
    doc_occurrences[i] = {}

    try:
        f = open(fname, "r", encoding="utf-8")
        contents = f.read().strip().lower()
        doc_occurrences[i]["contents"] = contents

        for topic, kw_list in keywords.items():
            if filt_crit([ kw for kw in kw_list if kw in contents ], kw_list):
            # if any([ kw in contents for kw in kw_list ]):
                doc_occurrences[i][topic] = sum([contents.count(x) for x in kw_list])
        
        f.close()
    except Exception as e:
        print(f"Err {fname}: {e}")
        pass

# Write out topics
topics_counts = { x: [] for x in keywords.keys() }
topics_medians = { x: 0.0 for x in keywords.keys() }
for i in doc_occurrences:
    for t in keywords:
        topics_counts[t].append(doc_occurrences[i][t])

for t in topics_medians:
    topics_medians[t] = numpy.median(topics_counts[t])

for i in doc_occurrences:
    
    topics = ["*mapaie"]

    for t in keywords:
        # if doc_occurrences[i][t] > topics_medians[t]:
        filt_crit = [ x in doc_occurrences[i]["contents"] for x in keywords[t] ] 
        # doc_has_topic = any(filt_crit)
        doc_has_topic = len([x for x in filt_crit if x])/len(filt_crit) >= 0.6

        if doc_has_topic:
            topics.append(f"*{t}")
            doc_counts[t] += 1
    
    # Corpus Iramuteq
    if iramuteq:
        print("**** " + " ".join(topics), file=corpus_file)
        print(doc_occurrences[i]["contents"], file=corpus_file)

    # Corpus cortext
    if cortext:
        for t in topics:
            # Creer dir topics
            if t.strip("*") != "mapaie":
                file = open(f"{t.strip('*')}/{i}.txt", "w", encoding="utf-8")
                print(doc_occurrences[i]["contents"], file=file) 

log_fp.close()
corpus_file.close()
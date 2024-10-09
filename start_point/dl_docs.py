import requests
import sys
import os
from random import choice
from tqdm import tqdm
import csv

requests.packages.urllib3.disable_warnings()

URL_FILE = "list_urls.txt"
MANIFESTOS_FILE = "all_manifestos.csv"
UA_FILE = "user_agents.txt"
OUT_FOLDER = "./docs"
LOG_FILE = "dl_docs.log"
log_fp = open(LOG_FILE, "w")


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
list_of_urls = [ x["URL"] for x in manifestos_list if x["Status"].lower() == "included" ]
user_agents = [ x.strip() for x in open(UA_FILE).readlines() ] 

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

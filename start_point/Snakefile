rule corpus:
    input:
        "preprocessed/",
    output:
        "corpus.txt"
    shell:
        "python create_corpus.py preprocessed/"

rule preprocess:
    input:
        "txts/",
    output:
        directory("preprocessed/")
    shell:
        "python preprocess.py"

rule parse:
    input:
        "docs/",
    output:
        directory("txts/")
    shell:
        "python parse_docs.py"

rule download:
    output:
        directory("docs/")
    shell:
        "python dl_docs.py"

rule clean:
    shell:
        "rm  -rf docs txts preprocessed"

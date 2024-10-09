import sys
from PyPDF2 import PdfReader
import magic
from bs4 import BeautifulSoup
from pathlib import Path


class Parser:

    def __init__(self, log_file=None):
        self.log_file = log_file

    ## To parse HTML files
    def parse_html(self, fname):
        # Read and parse html
        if "iso" in magic.from_file(fname).lower():
            charset ="iso-8859-1"
        else:
            charset = "utf-8"

        try:
            f_contents = open(fname, encoding=charset).read()
            contents = BeautifulSoup(f_contents, features="html.parser")
        except Exception as e:
            print(fname)
            print(magic.from_file(fname))
            print(e)
            sys.exit(2)

        all_children = list(contents.children)
        global MAX_CC
        global THE_CONTENT
        global MAX_DEPTH
        MAX_CC = 0
        MAX_DEPTH = 0

        def call(children, depth, len_content=0):
            """
                This function recursively explores all children (ie. performs a depth
                first traversal of the DOM tree), and finds the largest textual content
                that is not embedded in a script tag.
            """
            global MAX_CC
            global MAX_DEPTH
            global THE_CONTENT

            for child in children:

                if hasattr(child, "children") and child.name != "script":
                    # if element has not children, it is a leaf
                    if len(child.text) > MAX_CC:
                        MAX_DEPTH = depth + 1
                        MAX_CC = len(child.text)
                        THE_CONTENT = { "text": child.text, "tag": child.name }

                    # call on children elements (ie. go deeper in DOM tree)
                    call(child.children, depth + 1, len_content=len(child.text))
            
        # Initial call
        call(all_children, 0)

        ## Write to file
        txt_file = open(f"txts/{Path(fname).stem}.txt", "w+", encoding="utf-8")
        print(THE_CONTENT["text"], file=txt_file)
        txt_file.close()
        
        return;
    #####
    ### Parse PDFs
    def parse_pdf(self, fname):
        try:
            f = open(fname, "rb")
            reader = PdfReader(f)
            words = set()
            txt_file = open(f"txts/{Path(fname).stem}.txt", "w+", encoding="utf-8")
            
            for page in reader.pages:
                page_contents = page.extract_text()
                page_contents = page_contents.replace("-\n", "")
                page_contents = page_contents.replace("\n", " ")
                print(page_contents, file=txt_file)
                words = words.union(set(page_contents.split(" ")))
            
            f.close()
            txt_file.close()
            print(fname, len(words), file=self.log_file)
        except Exception as e:
            print(f"Err {fname}: {e}", file=self.log_file)
            pass
    #### END PARSER CODE

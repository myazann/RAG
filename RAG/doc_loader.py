from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader
import re

class DocumentLoader():

    def __init__(self, doc_name):

        self.doc_name = doc_name
        self.doc_type = doc_name.split(".")[-1]
        self.loader = None

    def load_doc(self, pdf_loader="unstructured"):

        if self.doc_type == "pdf":
            self.loader = self.pdf_loaders()[pdf_loader](self.doc_name)
        
        elif self.doc_type == "txt":
            self.loader = TextLoader(self.doc_name)
        
        try:
            doc = self.loader.load()
            return doc
        except:
            raise Exception("File doesn't exist or No loader yet for the file extension!")
        
    def pdf_loaders(self):

        return {
            "structured": PyPDFLoader,
            "unstructured": UnstructuredPDFLoader,
        }

    def trim_doc(self, doc):
    
        for page in doc:
            page.page_content = re.sub(r'\n+', '\n', page.page_content) 
            page.page_content = re.sub(r'\s{2,}', ' ', page.page_content)

        return doc
import re

from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader, TelegramChatFileLoader
from langchain.utilities import SQLDatabase

class FileLoader():

    def load(self, doc_name, pdf_loader="unstructured"):

        file_type = self.get_file_type(doc_name)
        if file_type == "db":
            print("Reading database!")
            doc = SQLDatabase.from_uri(f"sqlite:///{doc_name}") 
        else:
            if "telegram" in doc_name:
                loader = TelegramChatFileLoader(doc_name)
            elif file_type == "pdf":
                loader = self.pdf_loaders()[pdf_loader](doc_name)
            elif file_type == "txt":
                loader = TextLoader(doc_name)
            doc = loader.load()

        return doc

    def get_file_type(self, file_name):
        return file_name.split(".")[-1]

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
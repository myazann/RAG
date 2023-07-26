from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader

from configparser import ConfigParser

class DocumentLoader():

    def __init__(self, doc_name):

        self.doc_name = doc_name
        self.doc_type = doc_name.split(".")[-1]
        self.loader = self.get_loader()

    def get_loader(self):

        if self.doc_type == "pdf":

            parser = ConfigParser()
            parser.read("cfgs/doc_loader.cfg")
            def_pdf_loader = parser.get("pdf", "default")

            return self.pdf_loaders()[def_pdf_loader](self.doc_name)
        
        elif self.doc_type == "txt":
            return TextLoader(self.doc_name)
        
        else:
            raise Exception("No loader yet for the file extension, maybe it is incorrect?")
        
    def pdf_loaders(self):

        return {
            "structured": PyPDFLoader,
            "unstructured": UnstructuredPDFLoader,
        }

    def load_doc(self):
    
        doc = self.loader.load()
        # doc[0].page_content = doc[0].page_content.replace("\n\n", "\n")

        return doc
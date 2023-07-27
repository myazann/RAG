from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader

from utils import get_cfg_params

class DocumentLoader():

    def __init__(self, doc_name, *args):

        self.doc_name = doc_name
        self.doc_type = doc_name.split(".")[-1]
        self.cfg_params = get_cfg_params()["doc_loader"]
        self.loader = self.get_loader(*args)

    def get_loader(self, pdf_loader=None):

        if self.doc_type == "pdf":

            pdf_loader = self.cfg_params["default_pdf_loader"] if pdf_loader is None else pdf_loader
            return self.pdf_loaders()[pdf_loader](self.doc_name)
        
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
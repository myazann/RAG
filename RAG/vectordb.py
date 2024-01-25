from langchain_community.vectorstores import Chroma
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.storage import LocalFileStore

from RAG.utils import get_device

class VectorDB:
    def __init__(self, file_loader, embedding_function="hf_bge"):
        self.indb_files = []
        self.file_loader = file_loader
        self.embedding_function = self.get_embed_func(embedding_function)
        self.vector_db =  Chroma(embedding_function=self.embedding_function)

    def query_db(self):
        return self.vector_db.get()

    def get_embed_func(self, type):
        if type == "hf_bge":
            model_name = "BAAI/bge-base-en"
            model_kwargs = {"device": get_device()}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            fs = LocalFileStore("./embed_cache/")
            cached_embedder = CacheBackedEmbeddings.from_bytes_store(embeddings, fs, namespace=embeddings.model_name)
            return cached_embedder

    def add_file_to_db(self, file_name, web_search):
        files = self.file_loader.load(file_name, web_search)
        text_chunks, sources, file_types = self.file_loader.get_processed_texts(files)
        for chunk, source, file_type in zip(text_chunks, sources, file_types):
            self.indb_files.append(source)
            self.vector_db.add_texts(texts=[chunk], metadatas=[{"source": source, "file_type": file_type}])
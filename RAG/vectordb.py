import hashlib

from langchain_community.vectorstores import Chroma
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.storage import LocalFileStore

from RAG.utils import get_device

class VectorDB:
    def __init__(self, file_loader, embedding_function="hf_bge"):
        self.file_loader = file_loader
        self.embedding_function = self.get_embed_func(embedding_function)
        self.vector_db =  Chroma(embedding_function=self.embedding_function, persist_directory="./chroma_db")

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
        text_chunks, sources = self.file_loader.get_processed_texts(files)
        splitter_params = self.file_loader.splitter_params
        ids = [hashlib.sha256(f"{source}-chunksize:{splitter_params['chunk_size']}-chunkoverlap:{splitter_params['chunk_overlap']}-{i}".encode()).hexdigest()
               for i, source in enumerate(sources)]
        sources = [{"source": i} for i in sources]
        self.vector_db.add_texts(ids=ids, texts=text_chunks, metadatas=sources)
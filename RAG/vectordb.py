import hashlib
import os

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

class VectorDB:
    def __init__(self, file_loader, embedding_function="hf"):
        self.file_loader = file_loader
        self.embedding_function = self.get_embed_func(embedding_function)
        self.client = chromadb.Client()
        self.vector_db = self.client.get_or_create_collection(name="my_collection", embedding_function=self.embedding_function, metadata={"hnsw:space": "cosine"})

    def query_db(self, query=None, k=5, distance_threshold=0.5):
        if query:
            retr_docs = self.vector_db.query(query_texts=query, n_results=k)
            len_filt_docs = len([v for v in retr_docs["distances"][0] if v <= distance_threshold])
            return retr_docs["documents"][0][:len_filt_docs], retr_docs["distances"][0][:len_filt_docs], retr_docs["metadatas"][0][:len_filt_docs]
        else:
            return self.vector_db.get()

    def get_embed_func(self, type, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if type == "hf":
            return embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=os.getenv("HF_API_KEY"),
            model_name="sentence-transformers/use-cmlm-multilingual"
        )
        elif type == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002"
            )

    def add_file_to_db(self, file_name):
        files = self.file_loader.load(file_name)
        text_chunks, sources = self.file_loader.get_processed_texts(files)
        splitter_params = self.file_loader.splitter_params
        ids = [hashlib.sha256(f"{source}-chunksize:{splitter_params['chunk_size']}-chunkoverlap:{splitter_params['chunk_overlap']}-{i}".encode()).hexdigest()
               for i, source in enumerate(sources)]
        sources = [{"source": i} for i in sources]
        self.vector_db.upsert(ids=ids, documents=text_chunks, metadatas=sources)
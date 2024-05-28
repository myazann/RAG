import hashlib
import os
import time

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb.utils.embedding_functions as embedding_functions

class VectorDB:
    def __init__(self, file_loader, collection_name=None, embed_type="hf", hf_embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.file_loader = file_loader
        self.embedding_function = self.get_embed_func(embed_type, hf_embed_model)
        self.client = chromadb.Client()
        collection_name = collection_name if collection_name is not None else f"c_{round(time.time())}"
        self.vector_db = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_function, metadata={"hnsw:space": "cosine"})

    def query_db(self, query=None, k=5, distance_threshold=0.5):
        if query:
            retr_docs = self.vector_db.query(query_texts=query, n_results=k)
            len_filt_docs = len([v for v in retr_docs["distances"][0] if v <= distance_threshold])
            return retr_docs["documents"][0][:len_filt_docs], retr_docs["distances"][0][:len_filt_docs], retr_docs["metadatas"][0][:len_filt_docs]
        else:
            return self.vector_db.get()

    def get_embed_func(self, embed_type, hf_embed_model):
        if embed_type == "hf":
            return embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=os.getenv("HF_API_KEY"),
            model_name=hf_embed_model
        )
        elif embed_type == "openai":
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-large"
            )
        elif embed_type == "turkish":
            return TurkishEmbeddings()

    def add_file_to_db(self, file_name):
        files = self.file_loader.load(file_name)
        text_chunks, sources = self.file_loader.get_processed_texts(files)
        splitter_params = self.file_loader.splitter_params
        ids = [hashlib.sha256(f"{source}-chunksize:{splitter_params['chunk_size']}-chunkoverlap:{splitter_params['chunk_overlap']}-{i}".encode()).hexdigest()
               for i, source in enumerate(sources)]
        sources = [{"source": i} for i in sources]
        if ids:
            self.vector_db.upsert(ids=ids, documents=text_chunks, metadatas=sources)

class TurkishEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
        embeddings = model.encode(input)
        embeddings_as_list = [embedding.tolist() for embedding in embeddings]
        return embeddings_as_list
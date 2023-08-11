import numpy as np

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter, DocumentCompressorPipeline
from langchain.retrievers.multi_query import MultiQueryRetriever

class Retriever():

    def __init__(self, database, fetch_k=20) -> None:

        self.database = database
        self.fetch_k = fetch_k
        self.base_retriever = None
        self.comp_retriever = None
        self.mq_retriever = None
        self.filters = []

    def init_base_retriever(self, search_type="mmr", k=5):
        fetch_k = k if k >= self.fetch_k else self.fetch_k
        self.base_retriever = self.database.as_retriever(search_type=search_type, search_kwargs={"k":k, "fetch_k":fetch_k})
    
    def add_embed_filter(self, embeddings, similarity_threshold=0.2):

        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)
        return self.add_filter(embeddings_filter)

    def add_doc_compressor(self, llm):

        llm_chain_extractor = LLMChainExtractor.from_llm(llm)
        return self.add_filter(llm_chain_extractor)
    
    def empty_filters(self):

        self.filters = []
        self.init_comp_retriever()

        return self.comp_retriever
    
    def add_filter(self, filter=None):

        self.filters.append(filter)
        self.init_comp_retriever()

        return self.comp_retriever
    
    def init_comp_retriever(self):

        pipeline_compressor = DocumentCompressorPipeline(transformers=self.filters)
        self.comp_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=self.base_retriever)

    def init_mq_retriever(self, llm, prompt, search_type="mmr", k=5):
        self.mq_retriever = MultiQueryRetriever.from_llm(retriever=self.database.as_retriever(search_type=search_type, search_kwargs={"k": k}),
                                                         llm=llm, prompt=prompt)

    def find_ideal_k(self, chatbot, chunk):

        chunk_len = chatbot.count_tokens(chunk)
        k = int(chatbot.context_length)/chunk_len
        return int(np.floor(k))
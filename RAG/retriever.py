from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

class Retriever():

    def __init__(self, database, type="comp", search_type="mmr", k="5", fetch_k=50, comp_pipe=None, llm=None, mq_prompt=None) -> None:
        self.database = database
        self.fetch_k = fetch_k
        self.type = type
        self.retriever = self.init_retriever(search_type, k, comp_pipe, llm, mq_prompt)

    def init_retriever(self, search_type, k, comp_pipe, llm, prompt):
        fetch_k = k if k >= self.fetch_k else self.fetch_k
        base_retriever = self.database.as_retriever(search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k})    
        if self.type == "base":
            return base_retriever
        elif self.type == "comp":
            if comp_pipe is None:
                print("No compression pipeline found, initializing retriever from the vector database!")
                return base_retriever
            else:
                return ContextualCompressionRetriever(base_compressor=comp_pipe, base_retriever=base_retriever)
        elif self.type == "multiquery":
            return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm, prompt=prompt)

    def get_docs(self, query):
        return self.retriever.invoke(query)
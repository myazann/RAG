import numpy as np
import os
import time

from RAG.prompter import Prompter
from RAG.vectordb import VectorDB
from RAG.file_loader import FileLoader
from RAG.output_formatter import query_reform_formatter

class Orchestrator():
    def __init__(self, llm, helper_llm=None):
        self.llm = llm
        self.helper_llm = helper_llm if helper_llm else llm
        self.prompter = Prompter()
        self.file_loader = FileLoader()
        self.db = VectorDB(self.file_loader)

    def decide_on_query(self, query, chat_history, num_iter=3):
        is_query = []
        query_gen_prompt = self.prompter.query_gen_prompt_claude(user_input=query)
        threshold = (num_iter//2)+1 
        for _ in range(num_iter):
            is_query.append(query_reform_formatter(self.helper_llm.prompt_chatbot(query_gen_prompt, chat_history).strip()))
        no_q_count = len([q for q in is_query if "NO QUERY" in q])
        if no_q_count > threshold:
            return None
        else:
            return list(set([q for q in is_query if "NO QUERY" not in q]))[0]
        
    def find_best_k(self, chunks, strategy="optim"):
        avg_chunk_len = np.mean([self.llm.count_tokens(c) for c in chunks])
        avail_space = 2*int(self.context_length)//3
        if strategy == "max":
            pass
        elif strategy == "optim":
            avail_space /= 2
        return int(np.floor(avail_space/avg_chunk_len))
    
    def handle_query(self, query, chat_history, rag_config):
        reform_query = ""
        context = None
        start_time = time.time()

        if self.file_loader.get_file_type(query) in ["pdf", "git", "url"]:
            if not os.path.exists(query):
                print("Could not find file!")
                return
            self.db.add_file_to_db(query)
            reform_query = query
            print(f"Time passed processing file: {round(time.time() - start_time, 2)} secs")
        else:
            if rag_config.web_search:
                reform_query = self.decide_on_query(query, chat_history)
            if reform_query:
                print(reform_query)
                search_urls = self.file_loader.web_search(reform_query)
                print(search_urls)
                self.db.add_file_to_db(search_urls)
                print(f"Time passed in web search: {round(time.time() - start_time, 2)} secs")

            all_db_docs = self.db.query_db()["documents"]
            if all_db_docs:
                k = 10
                if not reform_query:
                    reform_query = self.decide_on_query(query)
                if reform_query:
                    context, distances, _ = self.db.query_db(query=reform_query, k=k, distance_threshold=0.75)
                    print(distances)
                    print(f"Time passed in retrieval: {round(time.time() - start_time, 2)} secs")

            conv_chain_prompt = self.prompter.conv_agent_prompt()
            prompt_params = {"query": query, "context": context}
            print(f"Time passed until generation: {round(time.time() - start_time, 2)} secs!")
            response = self.llm.prompt_chatbot(conv_chain_prompt, prompt_params, chat_history).strip()
            print("\nChatbot:")
            self.llm.stream_output(response)
            end_time = time.time()
            print(f"\nTook {round(end_time - start_time, 2)} secs!\n")
            return response
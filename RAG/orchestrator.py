import numpy as np
import os
import time

from RAG.prompter import Prompter
from RAG.vectordb import VectorDB
from RAG.file_loader import FileLoader
from RAG.output_formatter import query_reform_formatter

class Orchestrator():
    def __init__(self, helper_llm=None):
        self.helper_llm = helper_llm
        self.prompter = Prompter()
        self.file_loader = FileLoader()
        self.db = VectorDB(self.file_loader)

    def decide_on_query(self, query, chat_history, llm=None, num_iter=3):
        if not llm:
            llm = self.helper_llm
        is_query = []
        query_gen_prompt = self.prompter.query_gen_prompt_claude(user_input=query)
        threshold = (num_iter//2)+1 
        for _ in range(num_iter):
            is_query.append(query_reform_formatter(llm.prompt_chatbot(query_gen_prompt, chat_history).strip()))
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
            
    def handle_query(self, llm, query, rag_config, chat_history=[]):
        reform_query = None
        context = None
        conv_agent_prompt = self.prompter.conv_agent_prompt(query=query, context=context)

        if self.file_loader.get_file_type(query) in ["pdf", "git", "url"]:
            if not os.path.exists(query):
                print("Could not find file!")
                return
            self.db.add_file_to_db(query)
            reform_query = query
        else:
            if rag_config.web_search:
                reform_query = self.decide_on_query(query, chat_history)
            if reform_query:
                print(reform_query)
                search_urls = self.file_loader.web_search(reform_query)
                print(search_urls)
                self.db.add_file_to_db(search_urls)

            all_db_docs = self.db.query_db()["documents"]
            if all_db_docs:
                k = 10
                if not reform_query:
                    reform_query = self.decide_on_query(query)
                if reform_query:
                    context, distances, _ = self.db.query_db(query=reform_query, k=k, distance_threshold=0.75)
                    print(distances)
            
            if context:
                context = llm.prepare_context(conv_agent_prompt, context, chat_history)
                conv_agent_prompt = self.prompter.conv_agent_prompt(query=reform_query, context=context)
            response = llm.prompt_chatbot(conv_agent_prompt, chat_history, stream=True).strip()
            return response
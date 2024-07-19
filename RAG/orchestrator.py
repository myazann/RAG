import numpy as np

from RAG.output_formatter import query_reform_formatter

class Orchestrator():
    def __init__(self, llm):
        self.llm = llm

    def decide_on_query(self, prompt, num_iter=3):
        is_query = []
        threshold = (num_iter//2)+1 
        for _ in range(num_iter):
            is_query.append(query_reform_formatter(self.llm.prompt_chatbot(prompt).strip()))
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
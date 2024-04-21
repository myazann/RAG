import configparser
import os
from pathlib import Path
import sys
import time
import huggingface_hub

import tiktoken
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic

from RAG.output_formatter import query_reform_formatter

def get_model_cfg():
    config = configparser.ConfigParser()
    config.read(os.path.join(Path(__file__).absolute().parent, "model_config.cfg"))
    return config

def choose_bot(model_name=None, model_params=None, gen_params=None):
    while True: 
        if model_name is None:
            model_cfg = get_model_cfg()
            models = model_cfg.sections()
            model_families = dict({str(k): v for k, v in enumerate(sorted(set([model.split("-")[0] for model in models])))})
            print("Here are the available model families, please choose one:\n")
            for i, repo in model_families.items():
                print(f"{i}: {repo}")  
            while True:
                model_family_id = input()
                model_family = model_families.get(model_family_id)
                if model_family is None:
                    print("Please select from one of the options!")
                else:
                    break
            num_repo = dict({str(k): v for k, v in enumerate([model for model in models if model_family in model])})
            print("\nChoose a version or type 'b' to return to the previous menu:\n")
            for i, repo in num_repo.items():
                repo_name = repo.replace("_", "-")
                print(f"{i}: {repo_name}")  
            while True:
                model_id = input()
                if model_id.lower() == 'b':
                    print("Returning to model family selection...\n")
                    break
                model_name = num_repo.get(model_id)
                if model_name is None:
                    print("Please select from one of the options!")
                else:
                    return Chatbot(model_name, model_params, gen_params)
        else:
            return Chatbot(model_name, model_params, gen_params)

class Chatbot:

    def __init__(self, model_name, model_params=None, gen_params=None) -> None:
        huggingface_hub.login(token=os.getenv("HF_API_KEY"), new_session=False)
        self.cfg = get_model_cfg()[model_name]
        self.model_name = model_name
        self.family = model_name.split("-")[0]
        self.repo_id = self.cfg.get("repo_id")
        self.context_length = self.cfg.get("context_length")
        self.model_type = self.get_model_type()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.get_model_params(model_params)
        self.gen_params = self.get_gen_params(gen_params)
        self.model = self.init_model()

    def prompt_chatbot(self, prompt, chat_history=[]):
        if chat_history:
            chat_history = self.trunc_chat_history(chat_history)
        if self.model_type in ["default", "AWQ", "GPTQ"]:
            if self.family in ["MISTRAL", "GEMMA"]:
                prompt = [
                    {
                        "role": "user",
                        "content": f"{self.get_bot_info()}\n{prompt[0]['content']}\n{prompt[1]['content']}"
                    },
                    ]
                message = chat_history + prompt
            else:
                self_info = [
                    {
                        "role": "system",
                        "content": f"{self.get_bot_info()}"
                    },
                    ]
                message = self_info + [prompt[0]] + chat_history + [prompt[1]]
            pipe = pipeline("conversational", model=self.model, tokenizer=self.tokenizer, **self.gen_params)
            return pipe(message).messages[-1]["content"]
        elif self.model_type in ["PPLX", "GROQ"] or self.family == "CHATGPT":
            self_info = [
                    {
                        "role": "system",
                        "content": f"{self.get_bot_info()}"
                    },
                    ]
            message = self_info + [prompt[0]] + chat_history + [prompt[1]]
            response = self.model.chat.completions.create(model=self.repo_id, messages=message, **self.gen_params)
            return response.choices[0].message.content
        elif self.family == "CLAUDE":
            sys_msg = f"{self.get_bot_info()}\nprompt[0]['content']"
            message = chat_history + [
                    {
                        "role": "user",
                        "content": f"{prompt[1]['content']}"
                    },
                    ]
            response = self.model.messages.create(model=self.repo_id, messages=message, system=sys_msg, **self.gen_params)
            return response.content[0].text
    
    def stream_output(self, output):
        for char in output:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.005)
    
    def count_tokens(self, prompt):
        if isinstance(prompt, list):
            prompt = f"{prompt[0]['content']}\n{prompt[1]['content']}"
        if self.family == "CHATGPT":
            encoding = tiktoken.encoding_for_model(self.repo_id)
            return len(encoding.encode(prompt))
        elif self.family == "CLAUDE":
            return self.model.count_tokens(prompt)
        else:
            return len(self.tokenizer(prompt).input_ids)

    def get_bot_info(self):
        return f"Your name is {self.family.capitalize()}. You are named after the AI model that powers you."
    
    def trunc_chat_history(self, chat_history):
        hist_dedic_space = int(self.context_length)//8
        total_hist_tokens = sum(self.count_tokens(tm['content']) for tm in chat_history)
        while total_hist_tokens > hist_dedic_space:
            removed_message = chat_history.pop(0)
            total_hist_tokens -= self.count_tokens(removed_message['content'])
        return chat_history

    def find_best_k(self, chunks, strategy="optim"):
        avg_chunk_len = np.mean([self.count_tokens(c) for c in chunks])
        avail_space = 2*int(self.context_length)//3
        if strategy == "max":
            pass
        elif strategy == "optim":
            avail_space /= 2
        return int(np.floor(avail_space/avg_chunk_len))

    def get_model_type(self):
        if self.model_name.endswith("GPTQ"):
            return "GPTQ"
        elif self.model_name.endswith("AWQ"):
            return "AWQ"
        elif self.model_name.endswith("PPLX"):
            return "PPLX"
        elif self.model_name.endswith("GROQ"):
            return "GROQ"
        elif self.family in ["CLAUDE", "CHATGPT"]:
            return "proprietary"
        else:
            return "default"
        
    def init_tokenizer(self):
        if self.model_type in ["AWQ", "GPTQ", "PPLX", "GROQ"]:
            return AutoTokenizer.from_pretrained(self.cfg.get("tokenizer"), use_fast=True)
        elif self.model_type in ["proprietary"]:
            return None
        else:
            return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
            
    def get_gen_params(self, gen_params):
        name_token_var = "max_tokens" if self.model_type in ["PPLX", "GROQ", "proprietary"] else "max_new_tokens"
        if gen_params is None:
            return {name_token_var: 512}
        if "max_new_tokens" in gen_params and name_token_var != "max_new_tokens":
            gen_params[name_token_var] = gen_params.pop("max_new_tokens")
        elif "max_tokens" in gen_params and name_token_var != "max_tokens":
            gen_params[name_token_var] = gen_params.pop("max_tokens")
        return gen_params
    
    def default_model_params(self):
        if self.family == "LLAMA2":
            return {
                "trust_remote_code": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "token": True,
                }
        else:
            return {}
    
    def get_model_params(self, model_params):
        if model_params is None:
            if self.model_type == "PPLX":
                return {
                        "base_url": "https://api.perplexity.ai",
                        "api_key": os.getenv("PPLX_API_KEY")
                }
            elif self.model_type == "GROQ":
                return {
                    "api_key": os.getenv("GROQ_API_KEY")
                }
            elif self.family == "CLAUDE":
                return {
                    "api_key": os.getenv("ANTHROPIC_API_KEY")
                }
            elif self.family == "CHATGPT":
                return {
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            else:
                return self.default_model_params()
        else:
            return model_params
    
    def init_model(self):
        if self.family == "CLAUDE":
            return Anthropic(**self.model_params)
        elif self.family == "CHATGPT" or self.model_type == "PPLX":
            return OpenAI(**self.model_params)
        elif self.model_type == "GROQ":
            return Groq(**self.model_params)                   
        else:
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params,
                    low_cpu_mem_usage=True,
                    device_map="auto")
        
    def decide_on_query(self, prompt, num_iter=3):
        is_query = []
        threshold = (num_iter//2)+1 
        for _ in range(num_iter):
            is_query.append(query_reform_formatter(self.prompt_chatbot(prompt).strip()))
        is_q_count = len([q for q in is_query if "NO QUERY" in q])
        if is_q_count > threshold:
            return "NO QUERY"
        else:
            return [q for q in is_query if "NO QUERY" not in q]       
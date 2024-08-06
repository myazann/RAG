import configparser
import os
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import hf_hub_download, login
import tiktoken
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
logging.set_verbosity_error()
from openai import OpenAI
from llama_cpp import Llama
from groq import Groq
from anthropic import Anthropic
import google.generativeai as genai

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
            num_repo = dict({str(k): v for k, v in enumerate([model for model in models if model.startswith(model_family)])})
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
        login(token=os.getenv("HF_API_KEY"), new_session=False)
        self.cfg = get_model_cfg()[model_name]
        self.model_name = model_name
        self.family = model_name.split("-")[0]
        self.repo_id = self.cfg.get("repo_id")
        self.file_name = self.cfg.get("file_name", None)
        self.context_length = int(self.cfg.get("context_length"))
        self.model_type = self.get_model_type()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.get_model_params(model_params)
        self.gen_params = self.get_gen_params(gen_params)
        self.model = self.init_model()

    def prompt_chatbot(self, prompt, prompt_params=None, chat_history=[], stream=False):
        prompt = self.prepare_prompt(prompt, prompt_params, chat_history)
        
        if self.model_type in ["PPLX", "GROQ", "TGTR"] or self.family == "GPT":
            if len(prompt) > 1:
                message = [prompt[0]] + chat_history + [prompt[1]]
            else:
                message = chat_history + prompt
            response = self.model.chat.completions.create(model=self.repo_id, messages=message, **self.gen_params)
            response = response.choices[0].message.content
        elif self.family == "CLAUDE":
            if len(prompt) > 1:
                sys_msg = prompt[0]["content"]
                message = chat_history + [prompt[1]]
                response = self.model.messages.create(model=self.repo_id, messages=message, system=sys_msg, **self.gen_params)
            else:
                message = chat_history + prompt
                response = self.model.messages.create(model=self.repo_id, messages=message, **self.gen_params)
            response = response.content[0].text   
        elif self.family == "GEMINI":
            if len(prompt) > 1:
                message = [prompt[0]] + chat_history + [prompt[1]]
            else:
                message = chat_history + prompt
            messages = []
            for turn in message:
                role = "user" if turn["role"] in ["user", "system"] else "model"
                messages.append({
                    "role": role,
                    "parts": [turn["content"]]
                })
            response = self.model.generate_content(messages, generation_config=genai.types.GenerationConfig(**self.gen_params))
            response = response.text     
        else:
            if self.family in ["MISTRAL", "GEMMA"]:
                if len(prompt) > 1:
                    message = chat_history + [{"role": "user", "content": "\n".join([turn["content"] for turn in prompt])}]
                else:
                    message = chat_history + prompt
            else:
                if len(prompt) > 1:
                    message = [prompt[0]] + chat_history + [prompt[1]]
                else:
                    message = chat_history + prompt
            if self.model_type == "GGUF":
                response = self.model.create_chat_completion(message, **self.gen_params)
                response = response["choices"][-1]["message"]["content"]
            else:
                pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, **self.gen_params)
                response = pipe(message)[0]["generated_text"][-1]["content"]
        if stream:
            self.stream_output(response)
        return response
    
    def stream_output(self, output):
        for char in output:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.005)
    
    def count_tokens(self, prompt):
        if isinstance(prompt, list):
            prompt = "\n".join([turn["content"] for turn in prompt])
        if self.family == "GPT":
            encoding = tiktoken.encoding_for_model(self.repo_id)
            return len(encoding.encode(prompt))
        elif self.family == "GEMINI":
            return self.model.count_tokens(prompt).total_tokens
        elif self.family == "CLAUDE":
            return self.model.count_tokens(prompt)
        else:
            return len(self.tokenizer(prompt).input_ids)
    
    def trunc_chat_history(self, chat_history, hist_dedic_space=0.2):
        hist_dedic_space = int(self.context_length*0.2)
        total_hist_tokens = sum(self.count_tokens(tm['content']) for tm in chat_history)
        while total_hist_tokens > hist_dedic_space:
            removed_message = chat_history.pop(0)
            total_hist_tokens -= self.count_tokens(removed_message['content'])
        return chat_history
    
    def get_avail_space(self, prompt):
        avail_space = self.context_length - self.gen_params[self.name_token_var] - self.count_tokens(prompt)
        if avail_space <= 0:
            return None
        else:
            return avail_space 

    def prepare_prompt(self, prompt, prompt_params, chat_history=[]):
        if chat_history:
            chat_history = self.trunc_chat_history(chat_history)
        prompt = prompt(**prompt_params) + chat_history
        avail_space = self.get_avail_space(prompt)   
        if not avail_space:
            if "context" in prompt_params:
                print("Too much context, removing ")
            return "Sorry, I can't process that much text at the same time. Can you please shorten your message?"
        if avail_space:         
            while True:
                info = "\n".join([doc for doc in context])
                if self.count_tokens(info) > avail_space:
                    print("Context exceeds context window, removing one document!")
                    context = context[:-1]
                else:
                    break
            return info
        else:
            return -1

    def get_model_type(self):
        if self.model_name.endswith("AWQ"):
            return "AWQ"
        elif self.model_name.endswith("PPLX"):
            return "PPLX"
        elif self.model_name.endswith("GROQ"):
            return "GROQ"
        elif self.model_name.endswith("TGTR"):
            return "TGTR"
        elif self.model_name.endswith("GGUF"):
            return "GGUF"
        elif self.family in ["CLAUDE", "GPT", "GEMINI"]:
            return "proprietary"
        else:
            return "default"
        
    def init_tokenizer(self):
        if self.model_type in ["AWQ", "GPTQ", "PPLX", "GROQ", "GGUF", "TGTR"]:
            return AutoTokenizer.from_pretrained(self.cfg.get("tokenizer"), use_fast=True)
        elif self.model_type in ["proprietary"]:
            return None
        else:
            return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
            
    def get_gen_params(self, gen_params):
        if self.family == "GEMINI":
            self.name_token_var = "max_output_tokens"
        elif self.model_type in ["PPLX", "GROQ", "GGUF", "TGTR", "proprietary"]:
            self.name_token_var = "max_tokens"
        else:
            self.name_token_var = "max_new_tokens"
        if gen_params is None:
            return {self.name_token_var: 512}
        if "max_new_tokens" in gen_params and self.name_token_var != "max_new_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_new_tokens")
        elif "max_tokens" in gen_params and self.name_token_var != "max_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_tokens")
        elif "max_output_tokens" in gen_params and self.name_token_var != "max_output_tokens":
            gen_params[self.name_token_var] = gen_params.pop("max_output_tokens")
        return gen_params
    
    def get_model_params(self, model_params):
        if model_params is None:
            if self.model_type == "PPLX":
                return {
                    "base_url": "https://api.perplexity.ai",
                    "api_key": os.getenv("PPLX_API_KEY")
                }
            elif self.model_type == "GROQ":
                return {
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key": os.getenv("GROQ_API_KEY")
                }
            elif self.model_type == "TGTR":
                return {
                    "base_url": "https://api.together.xyz/v1",
                    "api_key": os.getenv("TOGETHER_API_KEY")
                }
            elif self.family == "CLAUDE":
                return {
                    "api_key": os.getenv("ANTHROPIC_API_KEY")
                }
            elif self.family == "GPT":
                return {
                    "api_key": os.getenv("OPENAI_API_KEY")
                }
            elif self.family == "GEMINI":
                return {
                    "api_key": os.getenv("GOOGLE_API_KEY")
                }
            elif self.model_type == "GGUF":
                return {
                    "n_gpu_layers": -1,
                    "verbose": False,
                    "n_ctx": self.context_length
                }
            else:
                return {}
        else:
            return model_params
    
    def init_model(self):
        if self.family == "CLAUDE":
            return Anthropic(**self.model_params)
        elif self.family == "GPT" or self.model_type in ["PPLX", "TGTR", "GROQ"]:
            return OpenAI(**self.model_params)       
        elif self.family == "GEMINI":
            genai.configure(**self.model_params)
            return genai.GenerativeModel(self.repo_id)       
        elif self.model_type == "GGUF":
            if os.getenv("HF_HOME") is None:
                hf_cache_path = os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "hub")
            else:
                hf_cache_path = os.getenv("HF_HOME")
            model_path = os.path.join(hf_cache_path, self.file_name)
            if not os.path.exists(model_path):
                hf_hub_download(repo_id=self.repo_id, filename=self.file_name, local_dir=hf_cache_path)
            return Llama(model_path=model_path, **self.model_params)
        else: 
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params,
                    low_cpu_mem_usage=True,
                    device_map="auto")
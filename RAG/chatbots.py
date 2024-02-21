import configparser
import os
from pathlib import Path
import urllib.request
import sys
import time

import tiktoken
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.chat_models import ChatAnthropic
from openai import OpenAI
from llama_cpp import Llama

def get_model_cfg():
    config = configparser.ConfigParser()
    config.read(os.path.join(Path(__file__).absolute().parent, "model_config.cfg"))
    return config

def choose_bot(model_name=None, model_params=None, gen_params=None, q_bits=None):
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
        print("\nChoose a version:\n")
        for i, repo in num_repo.items():
            repo_name = repo.replace("_", "-")
            print(f"{i}: {repo_name}")  
        while True:
            model_id = input()
            model_name = num_repo.get(model_id)
            if model_name is None:
                print("Please select from one of the options!")
            else:
                break
    return Chatbot(model_name, model_params, gen_params, q_bits)

class Chatbot:

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        self.cfg = get_model_cfg()[model_name]
        self.model_name = model_name
        self.family = model_name.split("-")[0]
        self.repo_id = self.cfg.get("repo_id")
        self.context_length = self.cfg.get("context_length")
        self.q_bit = q_bits
        self.model_type = self.get_model_type()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.get_model_params(model_params)
        self.gen_params = self.get_gen_params(gen_params)
        self.model = self.init_model()

    def prompt_chatbot(self, prompt):
        if self.model_type in ["default", "AWQ", "GPTQ"]:
            if self.family in ["MISTRAL", "GEMMA"]:
                prompt = [
                    {
                        "role": "user",
                        "content": f"{prompt[0]['content']}\n{prompt[1]['content']}"
                    },
                    ]
            pipe = pipeline("conversational", model=self.model, tokenizer=self.tokenizer, **self.gen_params)
            return pipe(prompt).messages[-1]["content"]
        elif self.family == "CHATGPT" or self.model_type == "PPLX":
            response = self.model.chat.completions.create(model=self.repo_id, messages=prompt, **self.gen_params)
            return response.choices[0].message.content
    
    def stream_output(self, output):
        def word_by_word_generator(text):
            for word in text.split():
                yield word        
        for chunk in word_by_word_generator(output):
              sys.stdout.write(chunk + ' ')
              sys.stdout.flush()
              time.sleep(0.02)
    
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

    def find_best_k(self, chunks, strategy="optim"):
        prompt_spc = 256
        avg_chunk_len = np.mean([self.count_tokens(c) for c in chunks])
        avail_space = int(self.context_length) - prompt_spc
        if strategy == "max":
            pass
        elif strategy == "optim":
            avail_space /= 2
        return int(np.floor(avail_space/avg_chunk_len))

    def get_model_type(self):
        if self.model_name.endswith("GPTQ"):
            return "GPTQ"
        elif self.model_name.endswith("GGUF") or self.repo_id.endswith("GGML"):
            return "GGUF"
        elif self.model_name.endswith("AWQ"):
            return "AWQ"
        elif self.model_name.endswith("PPLX"):
            return "PPLX"
        elif self.family in ["CLAUDE", "CHATGPT"]:
            return "proprietary"
        else:
            return "default"
        
    def init_tokenizer(self):
        if self.model_type in ["GGUF", "AWQ", "GPTQ", "PPLX"]:
            return AutoTokenizer.from_pretrained(self.cfg.get("tokenizer"), use_fast=True)
        elif self.model_type in ["proprietary"]:
            return None
        else:
            return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
            
    def get_gen_params(self, gen_params):
        if self.model_type in ["GGUF", "PPLX"] or self.family == "CHATGPT":
            name_token_var = "max_tokens"
        elif self.family == "CLAUDE":
            name_token_var = "max_tokens_to_sample"
        else:
            name_token_var = "max_new_tokens"
        if gen_params is None:
            return {
            name_token_var: 512,
            #"temperature": 0.7,
            }
        elif "max_new_tokens" or "max_tokens_to_sample" in gen_params.keys():
            value = gen_params.pop("max_new_tokens")
            gen_params[name_token_var] = value
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
            if self.model_type == "GGUF":
                rope_freq_scale = float(self.cfg.get("rope_freq_scale")) if self.cfg.get("rope_freq_scale") else 1
                return {
                    "n_gpu_layers": -1,
                    "n_batch": 512,
                    "verbose": False,
                    "n_ctx": self.context_length,
                    "rope_freq_scale": rope_freq_scale
                }
            elif self.model_type == "PPLX":
                return {
                        "base_url": "https://api.perplexity.ai",
                        "api_key": os.getenv("PPLX_API_KEY")
                }
            else:
                return self.default_model_params()
        else:
            return model_params
    
    def init_model(self):
        if self.family == "CLAUDE":
            return ChatAnthropic(model=self.repo_id, streaming=True, **self.gen_params)
        elif self.family == "CHATGPT" or self.model_type == "PPLX":
            return OpenAI(**self.model_params)
        elif self.model_type == "GGUF":
            if os.getenv("HF_HOME") is None:
                hf_cache_path = os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "transformers")
            else:
                hf_cache_path = os.getenv("HF_HOME")
            model_folder = os.path.join(hf_cache_path, self.repo_id.replace("/", "-"))
            bit_range = range(2, 9)
            if self.q_bit not in bit_range:
                print("This is a quantized model, please choose the number of quantization bits: ")
                for i in bit_range:
                    print(f"{i}")  
                while True:
                    q_bit = input()
                    if q_bit.isdigit():
                        if int(q_bit) not in bit_range:
                            print("Please select from one of the options!")
                        else:
                            self.q_bit = q_bit
                            break
                    else:
                        print("Please enter a number!")
            model_basename = "-".join(self.repo_id.split('/')[1].split("-")[:-1]).lower()
            if self.q_bit in [2, 6]:
                model_basename = f"{model_basename}.Q{self.q_bit}_K.gguf"
            elif self.q_bit == 8:
                model_basename = f"{model_basename}.Q{self.q_bit}_0.gguf"
            else:    
                model_basename = f"{model_basename}.Q{self.q_bit}_K_M.gguf"
            model_url_path = f"https://huggingface.co/{self.repo_id}/resolve/main/{model_basename}"
            if not os.path.exists(os.path.join(model_folder, model_basename)):
                os.makedirs(model_folder, exist_ok=True)
                try:
                    print("Downloading model!")
                    urllib.request.urlretrieve(model_url_path, os.path.join(model_folder, model_basename))
                except Exception as e:
                    print(e)
                    print("Couldn't find the model, please choose again! (Maybe the model isn't quantized with this bit?)")
            return Llama(
                    model_path=os.path.join(model_folder, model_basename),
                    **self.model_params,
                    **self.gen_params)     
        else:
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params,
                    low_cpu_mem_usage=True,
                    device_map="auto")
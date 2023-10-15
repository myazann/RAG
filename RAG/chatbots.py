import torch
import configparser
import os
from pathlib import Path
import urllib.request

from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.llms import LlamaCpp
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM

from RAG.output_formatter import strip_all

def get_model_cfg():

    config = configparser.ConfigParser()
    config.read(os.path.join(Path(__file__).absolute().parent, "model_config.cfg"))
    return config

def choose_bot(model_name=None, model_params=None, gen_params=None, q_bits=None):

    if model_name is None:

        model_cfg = get_model_cfg()
        models = model_cfg.sections()
        model_families = dict({str(k): v for k, v in enumerate(sorted(set([model.split("-")[0] for model in models ])))})
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

    if "FALCON" in model_name:
        return Falcon(model_name, model_params, gen_params, q_bits)
    elif "VICUNA" in model_name:
        return Vicuna(model_name, model_params, gen_params, q_bits)
    elif "LLAMA" in model_name:
        return LLaMA2(model_name, model_params, gen_params, q_bits)
    elif "BELUGA" in model_name:
        return StableBeluga(model_name, model_params, gen_params, q_bits)
    elif "LUNA" in model_name:
        return Luna(model_name, model_params, gen_params, q_bits)
    elif "GPT" in model_name:
        return ChatGPT(model_name, model_params, gen_params)
    elif "CLAUDE" in model_name:
        return Claude(model_name, model_params, gen_params)
    elif "MISTRAL" in model_name:
        return Mistral(model_name, model_params, gen_params, q_bits)
    else:
        print("Chatbot not implemented yet! (or it doesn't exist?)")

class Chatbot:

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:

        self.cfg = get_model_cfg()[model_name]
        self.name = model_name
        self.repo_id = self.cfg.get("repo_id")
        self.model_basename = self.cfg.get("basename")
        self.context_length = self.cfg.get("context_length")
        self.q_bit = str(q_bits)
        self.model_type = self.get_model_type()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.get_model_params(model_params)
        self.gen_params = self.get_gen_params(gen_params)
        self.model = self.init_model()
        self.pipe = self.init_pipe()

    def prompt_template(self):
        return None
    
    def count_tokens(self, prompt):
        if isinstance(prompt, str):
            return len(self.tokenizer(prompt).input_ids)
        if isinstance(prompt, list):
            return max([len(self.tokenizer(chunk).input_ids) for chunk in prompt])
        
    def get_model_type(self):
        if self.repo_id.endswith("GPTQ"):
            return "GPTQ"
        elif self.repo_id.endswith("GGUF") or self.repo_id.endswith("GGML"):
            return "GGUF"
        elif self.repo_id.endswith("AWQ"):
            return "AWQ"
        else:
            return "default"
        
    def init_tokenizer(self):
        if self.model_type == "GGUF":
            return AutoTokenizer.from_pretrained(self.cfg.get("tokenizer"), use_fast=True)
        elif "claude" in self.repo_id or "gpt" in self.repo_id:
            return None
        else:
            return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
            
    def get_gen_params(self, gen_params):

        if self.model_type == "GGUF" or "gpt" in self.repo_id:
            name_token_var = "max_tokens"
        elif "claude" in self.repo_id:
            name_token_var = "max_tokens_to_sample"
        else:
            name_token_var = "max_new_tokens"

        if gen_params is None:
            return {
            name_token_var: 512,
            "temperature": 0.7,
            }
        
        elif "max_new_tokens" or "max_tokens_to_sample" in gen_params.keys():
            value = gen_params.pop("max_new_tokens")
            gen_params[name_token_var] = value

        return gen_params

    def gptq_params(self):
        return {
                "device_map": "auto",
                "use_safetensors": True,
                "trust_remote_code": True,
                "use_triton": False,
                "quantize_config": None
                }
    
    def ggum_params(self):
        rope_freq_scale = float(self.cfg.get("rope_freq_scale")) if self.cfg.get("rope_freq_scale") else 1
        return {
                "n_gpu_layers": -1,
                "n_batch": 512,
                "verbose": False,
                "n_ctx": self.context_length,
                "rope_freq_scale": rope_freq_scale
                }
    
    def awq_params(self):
        return {
            "fuse_layers": True,
            "trust_remote_code": False,
            "safetensors": True
        }
    
    def default_model_params(self):
        return {}
    
    def get_model_params(self, model_params):
        if model_params is None:
            if self.model_type == "GPTQ":
                return self.gptq_params()
            elif self.model_type == "GGUF":
                return self.ggum_params()
            elif self.model_type == "AWQ":
                return self.awq_params()
            else:
                return self.default_model_params()
        else:
            return model_params
    
    def init_model(self):
        if self.model_type == "GPTQ":
            return AutoGPTQForCausalLM.from_quantized(
                    self.repo_id,
                    model_basename=self.model_basename,
                    **self.model_params)
        elif self.model_type == "AWQ":
            return AutoAWQForCausalLM.from_quantized(
                self.repo_id,
                **self.model_params)
        
        elif "claude" in self.repo_id:
            return ChatAnthropic(model=self.repo_id, **self.gen_params)
        
        elif "gpt" in self.repo_id:
            return ChatOpenAI(model=self.repo_id, **self.gen_params)
        
        elif self.model_type == "GGUF":
            if os.getenv("HF_HOME") is None:
                hf_cache_path = os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "transformers")
            else:
                hf_cache_path = os.getenv("HF_HOME")
            model_folder = os.path.join(hf_cache_path, self.repo_id.replace("/", "-"))

            bit_range = [str(i) for i in range(2, 9)]
            if self.q_bit not in bit_range:
                print("This is a quantized model, please choose the number of quantization bits: ")
                for i in bit_range:
                    print(f"{i}")  
                while True:
                    q_bit = input()
                    if q_bit not in bit_range:
                        print("Please select from one of the options!")
                    else:
                        self.q_bit = q_bit
                        break

            self.model_basename = "-".join(self.repo_id.split('/')[1].split("-")[:-1]).lower()
            if self.q_bit in ["2", "6"]:
                self.model_basename = f"{self.model_basename}.Q{self.q_bit}_K.gguf"
            elif self.q_bit == "8":
                self.model_basename = f"{self.model_basename}.Q{self.q_bit}_0.gguf"
            else:    
                self.model_basename = f"{self.model_basename}.Q{self.q_bit}_K_M.gguf"
            model_url_path = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.model_basename}"
            if not os.path.exists(os.path.join(model_folder, self.model_basename)):
                os.makedirs(model_folder, exist_ok=True)
                try:
                    print("Downloading model!")
                    urllib.request.urlretrieve(model_url_path, os.path.join(model_folder, self.model_basename))
                except Exception as e:
                    print(e)
                    print("Couldn't find the model, please choose again! (Maybe the model isn't quantized with this bit?)")

            return LlamaCpp(
                    model_path=os.path.join(model_folder, self.model_basename),
                    **self.model_params,
                    **self.gen_params)     
        else:
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params,
                    device_map="auto")
        
    def init_pipe(self):            
        if self.model_type == "GGUF" or "claude" in self.repo_id or "gpt" in self.repo_id:
            return self.model
        else:
            return HuggingFacePipeline(pipeline=pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, **self.gen_params))

class Vicuna(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return strip_all("""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user"s questions.
        USER: 
        {prompt}
        ASSISTANT:""")

class Falcon(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)
    
    def get_model_params(self):
        return {
                "torch_dtype": torch.bfloat16, 
                "trust_remote_code": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_return_sequences": 1,
                "do_sample": True,
                "top_k": 10,
                }
    
class LLaMA2(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return strip_all("""[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.<</SYS>>{prompt}[/INST]""")
    
    def default_model_params(self):
        return {
                "trust_remote_code": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_auth_token": True,
                }
    
class StableBeluga(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return strip_all("""### System: 
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.
        ### User: 
        {prompt}
        ### Assistant:""")        
    
class Luna(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return strip_all("""USER: {prompt}
        ASSISTANT:""")

class Claude(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None) -> None:
        super().__init__(model_name, model_params, gen_params)

    def prompt_template(self):
        return strip_all("""Human: {prompt}
        Assistant:""")
    
    def count_tokens(self, prompt):
        if isinstance(prompt, str):
            return self.model.count_tokens(prompt)
        if isinstance(prompt, list):
            return max([self.model.count_tokens(chunk) for chunk in prompt])
        
class ChatGPT(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None) -> None:
        super().__init__(model_name, model_params, gen_params)

    def prompt_template(self):
        return strip_all("""{prompt}""")
    
    def count_tokens(self, prompt):
        if isinstance(prompt, str):
            return self.model.get_num_tokens(prompt)
        if isinstance(prompt, list):
            return max([self.model.get_num_tokens(chunk) for chunk in prompt])
        
class Mistral(Chatbot):

    def __init__(self, model_name, model_params=None, gen_params=None, q_bits=None) -> None:
        super().__init__(model_name, model_params, gen_params, q_bits)

    def prompt_template(self):
        return strip_all("""<s>[INST] {prompt} [/INST]""")
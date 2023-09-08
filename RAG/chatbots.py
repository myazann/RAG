import torch
import configparser
import os
import subprocess
from pathlib import Path

from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, AutoConfig, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatAnthropic
from langchain.llms import LlamaCpp
from auto_gptq import AutoGPTQForCausalLM

from RAG.output_formatter import strip_all

def get_model_cfg():

    config = configparser.ConfigParser()
    config.read(os.path.join(Path(__file__).absolute().parent, "model_config.cfg"))
    return config

def choose_bot(device, model_name=None, gen_params=None):

    if model_name is None:

        model_cfg = get_model_cfg()
        models = model_cfg.sections()
        num_repo = dict({str(k): v for k, v in enumerate(models)})
        print("\nChoose a model from the list: (Use their number id for choosing)\n")
        for i, repo in num_repo.items():
            repo_name = repo.replace("_", "-")
            gpu_req = model_cfg[repo]['min_GPU_RAM']
            if int(gpu_req) == 0:
                print(f"{i}: {repo_name} (Doesn't need a GPU!)")  
            else:
                print(f"{i}: {repo_name} (Requires at least {gpu_req}GB of GPU RAM!)")  

        while True:
            model_id = input()
            model_name = num_repo.get(model_id)
            if model_name is None:
                print("Please select from one of the options!")
            else:
                break

    if "FALCON" in model_name:
        return Falcon(model_name, device, gen_params)
    elif "VICUNA" in model_name:
        return Vicuna(model_name, device, gen_params)
    elif "LLAMA" in model_name:
        return LLaMA2(model_name, device, gen_params)
    elif "MPT" in model_name:
        return MPT(model_name, device, gen_params)
    elif "BELUGA" in model_name:
        return StableBeluga(model_name, device, gen_params)
    elif "CLAUDE" in model_name:
        return Claude(model_name, gen_params)
    elif "LUNA" in model_name:
        return Luna(model_name, gen_params)
    else:
        print("Chatbot not implemented yet! (or it doesn't exist?)")

class Chatbot:

    def __init__(self, model_name, device, gen_params=None) -> None:

        self.cfg = get_model_cfg()[model_name]
        self.name = model_name
        self.repo_id = self.cfg.get("repo_id")
        self.model_basename = self.cfg.get("basename")
        self.context_length = self.cfg.get("context_length")
        self.device = device
        self.model_type = self.get_model_type()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.get_model_params()
        self.gen_params = self.get_gen_params() if gen_params is None else gen_params
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
        else:
            return "default"
        
    def init_tokenizer(self):
        if self.model_type == "GGUF":
            return AutoTokenizer.from_pretrained(self.cfg.get("tokenizer"), use_fast=True)
        else:
            return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
            
    def get_gen_params(self):
        name_token_var = "max_tokens" if self.model_type == "GGUF" else "max_new_tokens"
        return {
        name_token_var: 512,
        "temperature": 0.7,
    }

    def gptq_params(self):
        return {
                "device": self.device,
                "use_safetensors": True,
                "trust_remote_code": True,
                "use_triton": False,
                "quantize_config": None
                }
    
    def ggum_params(self):
        return {
                "n_gpu_layers": -1,
                "n_batch": 512,
                "verbose": False,
                "n_ctx": self.context_length
                }
    
    def default_model_params(self):
        return {}
    
    def get_model_params(self):
        if self.model_type == "GPTQ":
            return self.gptq_params()
        elif self.model_type == "GGUF":
            return self.ggum_params()
        else:
            return self.default_model_params()
    
    def init_model(self):
        if self.model_type == "GPTQ":
            return AutoGPTQForCausalLM.from_quantized(
                    self.repo_id,
                    model_basename=self.model_basename,
                    **self.model_params)
        elif self.model_type == "GGUF":
            if os.getenv("HF_HOME") is None:
                hf_cache_path = os.path.join(os.path.expanduser('~'), ".cache", "huggingface", "transformers")
            else:
                hf_cache_path = os.getenv("HF_HOME")
            model_folder = os.path.join(hf_cache_path, self.repo_id.replace("/", "-"))
            if not os.path.exists(os.path.join(model_folder, self.model_basename)):
                os.makedirs(model_folder, exist_ok=True)
                model_url_path = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.model_basename}"
                wget_command = f"wget -P {model_folder} {model_url_path}"
                subprocess.run(wget_command.split(), check=True)

            return LlamaCpp(
                    model_path=os.path.join(model_folder, self.model_basename),
                    **self.model_params,
                    **self.gen_params
                    )     
        else:
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params,
                    )
        
    def init_pipe(self):
        if self.model_type == "GGUF":
            return self.model
        else:
            return HuggingFacePipeline(pipeline=pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device, **self.gen_params))

class Vicuna(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        return strip_all("""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user"s questions.
        USER: 
        {prompt}
        ASSISTANT:""")

class MPT(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    def get_model_params(self):
        config = AutoConfig.from_pretrained(self.repo_id, trust_remote_code=True)
        config.init_device = self.device 
        config.max_seq_len = 8192

        return {
                #"config": config,
                "init_device": self.device,
                "max_seq_len": 4096,
                "torch_dtype": torch.bfloat16, 
                "trust_remote_code": True,
                }

    def get_gen_params(self):

        stop_token_ids = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
        
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False
        
        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        return {
                "return_full_text": True,
                "stopping_criteria": stopping_criteria, 
                "max_new_tokens": 512,  
                "repetition_penalty": 1.1
                }

class Falcon(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)
    
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

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        if "Yarn" in self.repo_id:
            return "{prompt}"
        else: 
            return strip_all("""[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.<</SYS>>{prompt}[/INST]""")
    
    def default_model_params(self):
        return {
                "trust_remote_code": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_auth_token": True,
                }
    
class StableBeluga(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        return strip_all("""### System: 
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.
        ### User: 
        {prompt}
        ### Assistant:""")        
    
class Luna(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        return strip_all("""USER: {prompt}
        ASSISTANT:""")

class Claude(Chatbot):

    def __init__(self, model_name, gen_params=None) -> None:

        self.cfg = get_model_cfg()[model_name]
        self.name = model_name
        self.repo_id = self.cfg.get("repo_id")
        self.model_basename = self.cfg.get("basename")
        self.context_length = self.cfg.get("context_length")
        self.gen_params = self.get_gen_params() if gen_params is None else self.reformat_params(gen_params)
        self.model_params = self.gen_params
        self.model = self.init_model()
        self.pipe = self.init_pipe()

    def prompt_template(self):
        return strip_all("""Human: {prompt}
        Assistant:""")
    
    def count_tokens(self, prompt):
        if isinstance(prompt, str):
            return self.model.count_tokens(prompt)
        if isinstance(prompt, list):
            return max([self.model.count_tokens(chunk) for chunk in prompt])
    
    def reformat_params(self, gen_params):
        if "max_new_tokens" in gen_params.keys():
            value = gen_params.pop("max_new_tokens")
            gen_params["max_tokens_to_sample"] = value
        return gen_params
    
    def get_gen_params(self):
        return {
            "max_tokens_to_sample": 512,
            "temperature": None
        }
    
    def init_model(self):
        return ChatAnthropic(model=self.repo_id, **self.gen_params)
    
    def init_pipe(self):
        return self.model
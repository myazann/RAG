import torch

from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, AutoConfig, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM

from enums import GPTQ_MODELNAMES, REPO_ID

def choose_bot(device, repo=None):
  
    if repo is None:
        repos = REPO_ID.__members__    
        repo_dict = dict((str(k), v) for k, v in enumerate(repos.keys()))

        print("\nChoose a model from the list: (Use their number id for choosing)\n")

        for i, repo in repo_dict.items():
            repo_name = repo.replace("_", "-")
            print(f"{i}: {repo_name}")  

        while True:

            model_id = input()
            repo_id = repo_dict.get(model_id)

            if repo_id is None:
                print("Please select from one of the options!")
            else:
                repo = repos[repo_id]
                break

    if "FALCON" in repo.name:
        return Falcon(repo, device)
    elif "VICUNA" in repo.name:
        return Vicuna(repo, device)
    elif "LLAMA" in repo.name:
        return LLaMA2(repo, device)
    elif "MPT" in repo.name:
        return MPT(repo, device)
    elif "GPT4ALL" in repo.name:
        return GPT4ALL(repo, device)  
    elif "BELUGA" in repo.name:
        return StableBeluga(repo, device)
    else:
        print("Chatbot not implemented yet! (or it doesn't exist?)")

class Chatbot:

    def __init__(self, repo, device) -> None:

        self.repo = repo
        self.device = device
        self.is_gptq = self.check_is_gptq()
        self.tokenizer = self.init_tokenizer()

        self.model_params = self.gptq_model_params() if self.is_gptq else self.get_model_params()
        self.gen_params = self.get_gen_params()
        self.model = self.init_model()
        self.pipe = self.init_pipe()

    def prompt_template(self):
        return None

    def get_gen_params(self):
        pass

    def get_model_params(self):
        pass

    def check_is_gptq(self):
        return True if "GPTQ" in self.repo.name else False
    
    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.repo.value, use_fast=True)
    
    def gptq_model_params(self):
        return {
                "device": self.device,
                "use_safetensors": True,
                "trust_remote_code": True,
                "use_triton": False,
                "quantize_config": None
                }
    
    def init_model(self):
        if self.is_gptq:
            return AutoGPTQForCausalLM.from_quantized(
                    self.repo.value,
                    model_basename=GPTQ_MODELNAMES[self.repo.name].value,
                    **self.model_params)
        else:
            return AutoModelForCausalLM.from_pretrained(
                    self.repo.value,
                    **self.model_params
                    )
        
    def init_pipe(self):
        return pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device, **self.gen_params)


class Vicuna(Chatbot):

    def __init__(self, repo, device) -> None:
        super().__init__(repo, device)

    def get_gen_params(self):
        return {
        "max_new_tokens": 512,
        "temperature": 0.7
    }

class GPT4ALL(Chatbot):

    def __init__(self, repo, device) -> None:
        super().__init__(repo, device)

    def get_gen_params(self):
        return {
        "max_new_tokens": 512,
        "temperature": 0.7
    }

class MPT(Chatbot):

    def __init__(self, repo, device) -> None:
        super().__init__(repo, device)

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    def get_model_params(self):
        config = AutoConfig.from_pretrained(self.repo.value, trust_remote_code=True)
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

    def __init__(self, repo, device) -> None:
        super().__init__(repo, device)
    
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

    def get_gen_params(self):
        return {
                "max_new_tokens": 512,
                "temperature": 0.7
                }
    
class LLaMA2(Chatbot):

    def __init__(self, repo, device) -> None:
        super().__init__(repo, device)

    def prompt_template(self, prompt):
        return f"""[INST] <<SYS>>
                  You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
                  If you don't know the answer to a question, please don't share false information.<</SYS>>
                  {prompt}[/INST]"""
    
    def get_model_params(self):
        return {
                "trust_remote_code": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_auth_token": True,
                }

    def get_gen_params(self):
        return {
                "max_new_tokens": 512,
                "temperature": 0.7
                } 
    
class StableBeluga(Chatbot):

    def __init__(self, repo, device) -> None:
        super().__init__(repo, device)

    def prompt_template(self, prompt):
        return f"""### System:
                       This is a system prompt, please behave and help the user.

                       ### User:
                       {prompt}

                       ### Assistant:
                       """

    def get_gen_params(self):
        return {
                "max_new_tokens": 512,
                "temperature": 0.7
                }     
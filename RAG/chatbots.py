import torch
import configparser

from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, AutoConfig, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatAnthropic
from auto_gptq import AutoGPTQForCausalLM

from RAG.utils import strip_all

def get_model_cfg():

    config = configparser.ConfigParser()
    config.read("model_config.cfg")
    return config

def choose_bot(device, model_name=None, gen_params=None):

    if model_name is None:

        models = get_model_cfg().sections()
        num_repo = dict({str(k): v for k, v in enumerate(models)})
        print("\nChoose a model from the list: (Use their number id for choosing)\n")
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
        return Falcon(model_name, device, gen_params)
    elif "VICUNA" in model_name:
        return Vicuna(model_name, device, gen_params)
    elif "LLAMA" in model_name:
        return LLaMA2(model_name, device, gen_params)
    elif "MPT" in model_name:
        return MPT(model_name, device, gen_params)
    elif "GPT4ALL" in model_name:
        return GPT4ALL(model_name, device, gen_params)  
    elif "BELUGA" in model_name:
        return StableBeluga(model_name, device, gen_params)
    elif "OPEN_CHAT" in model_name:
        return OpenChat(model_name, device, gen_params)
    elif "BTLM" in model_name:
        return BTLM(model_name, device, gen_params)
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
        self.is_gptq = self.check_is_gptq()
        self.tokenizer = self.init_tokenizer()
        self.model_params = self.gptq_model_params() if self.is_gptq else self.get_model_params()
        self.gen_params = self.get_gen_params() if gen_params is None else gen_params
        self.model = self.init_model()
        self.pipe = self.init_pipe()

    def prompt_template(self):
        return None

    def get_gen_params(self):
        return {}

    def get_model_params(self):
        return {}

    def check_is_gptq(self):
        return True if "GPTQ" in self.repo_id else False
    
    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
    
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
                    self.repo_id,
                    model_basename=self.model_basename,
                    **self.model_params)
        else:
            return AutoModelForCausalLM.from_pretrained(
                    self.repo_id,
                    **self.model_params
                    )
        
    def init_pipe(self):
        return HuggingFacePipeline(pipeline=pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device, **self.gen_params))

class Vicuna(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        return strip_all("""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user"s questions.
        USER: 
        {prompt}
        ASSISTANT:""")

    def get_gen_params(self):
        return {
        "max_new_tokens": 512,
        "temperature": 0.7
    }

class GPT4ALL(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def get_gen_params(self):
        return {
        "max_new_tokens": 512,
        "temperature": 0.7
    }

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

    def get_gen_params(self):
        return {
                "max_new_tokens": 512,
                "temperature": 0.7
                }
    
class LLaMA2(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        return strip_all("""[INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.<</SYS>>{prompt}[/INST]""")
    
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

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        return strip_all("""### System: 
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.
        ### User: 
        {prompt}
        ### Assistant:""")

    def get_gen_params(self):
        return {
                "max_new_tokens": 512,
                "temperature": 0.7
                }     
    
class OpenChat(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        return strip_all("""GPT4 User: {prompt}<|end_of_turn|>
        GPT4 Assistant:""")

    def get_gen_params(self):
        return {
                "max_new_tokens": 512,
                "temperature": 0.7
                }         
    
class BTLM(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def get_model_params(self):
        return {
                "trust_remote_code": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "max_length": 8192
                }

    def get_gen_params(self):
        return {
                "max_new_tokens": 512, 
                "repetition_penalty": 1.1,
                "do_sample": False, 
                "no_repeat_ngram_size": 2,
                "temperature": 0.7
                }    
    
class Luna(Chatbot):

    def __init__(self, model_name, device, gen_params=None) -> None:
        super().__init__(model_name, device, gen_params)

    def prompt_template(self):
        return strip_all("""USER: {prompt}
        ASSISTANT:""")

    def get_gen_params(self):
        return {
                "max_new_tokens": 512,
                "temperature": 0.7
                }     

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
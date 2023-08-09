import torch

from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, AutoConfig, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.chat_models import ChatAnthropic
from auto_gptq import AutoGPTQForCausalLM

from RAG.utils import strip_all

def get_repos():
    return {
        "MPT-7B": "mosaicml/mpt-7b-chat",
        "FALCON-7B": "tiiuae/falcon-7b-instruct",
        "GPT4ALL-13B-GPTQ": "TheBloke/GPT4All-13B-Snoozy-SuperHOT-8K-GPTQ",
        "VICUNA-7B-v1.5-GPTQ": "TheBloke/vicuna-7B-v1.5-GPTQ",
        "VICUNA-13B-v1.5-GPTQ": "TheBloke/vicuna-13B-v1.5-GPTQ",
        "LLAMA2-7B": "meta-llama/Llama-2-7b-chat-hf",
        "LLAMA2-7B-GPTQ": "TheBloke/Llama-2-7b-Chat-GPTQ",
        "LLAMA2-13B-GPTQ": "TheBloke/Llama-2-13B-chat-GPTQ",
        "STABLE-BELUGA-7B-GPTQ": "TheBloke/StableBeluga-7B-GPTQ",
        "STABLE-BELUGA-13B-GPTQ": "TheBloke/StableBeluga-13B-GPTQ",
        "OPEN-CHAT-GPTQ": "TheBloke/OpenChat-v3.2-GPTQ",
        "BTLM-3B-8K": "cerebras/btlm-3b-8k-base",
        "CLAUDE-V1": "claude-1.1",
        "CLAUDE-V2": "claude-2.0"
        }

def get_model_basenames():
    return {
        "GPT4ALL-13B-GPTQ": "gpt4all-snoozy-13b-superhot-8k-GPTQ-4bit-128g.no-act.order",
        "LLAMA2-7B-GPTQ": "gptq_model-4bit-128g",
        "LLAMA2-13B-GPTQ": "gptq_model-4bit-128g",
        "STABLE-BELUGA-7B-GPTQ": "gptq_model-4bit-128g",
        "STABLE-BELUGA-13B-GPTQ": "gptq_model-4bit-128g",
        "OPEN-CHAT-GPTQ": "gptq_model-4bit-128g"
    }

def choose_bot(device, repo_id=None, gen_params=None):
  
    if repo_id is None:

        num_repo = dict({str(k): v for k, v in enumerate(get_repos().keys())})
        print("\nChoose a model from the list: (Use their number id for choosing)\n")
        for i, repo in num_repo.items():
            repo_name = repo.replace("_", "-")
            print(f"{i}: {repo_name}")  

        while True:
            model_id = input()
            repo_id = num_repo.get(model_id)
            if repo_id is None:
                print("Please select from one of the options!")
            else:
                break

    if "FALCON" in repo_id:
        return Falcon(repo_id, device, gen_params)
    elif "VICUNA" in repo_id:
        return Vicuna(repo_id, device, gen_params)
    elif "LLAMA" in repo_id:
        return LLaMA2(repo_id, device, gen_params)
    elif "MPT" in repo_id:
        return MPT(repo_id, device, gen_params)
    elif "GPT4ALL" in repo_id:
        return GPT4ALL(repo_id, device, gen_params)  
    elif "BELUGA" in repo_id:
        return StableBeluga(repo_id, device, gen_params)
    elif "OPEN_CHAT" in repo_id:
        return OpenChat(repo_id, device, gen_params)
    elif "BTLM" in repo_id:
        return BTLM(repo_id, device, gen_params)
    elif "CLAUDE" in repo_id:
        return Claude(repo_id, gen_params)
    else:
        print("Chatbot not implemented yet! (or it doesn't exist?)")

class Chatbot:

    def __init__(self, repo_id, device, gen_params=None) -> None:

        self.repo = (repo_id, get_repos()[repo_id])
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
        return True if "GPTQ" in self.repo[1] else False
    
    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.repo[1], use_fast=True)
    
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
            basename = get_model_basenames().get(self.repo[0])
            if basename is None:
                    return AutoGPTQForCausalLM.from_quantized(
                        self.repo[1],
                        **self.model_params)
            else:
                return AutoGPTQForCausalLM.from_quantized(
                        self.repo[1],
                        model_basename=basename,
                        **self.model_params)
        else:
            return AutoModelForCausalLM.from_pretrained(
                    self.repo[1],
                    **self.model_params
                    )
        
    def init_pipe(self):
        return HuggingFacePipeline(pipeline=pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device, **self.gen_params))

class Vicuna(Chatbot):

    def __init__(self, repo, device, gen_params=None) -> None:
        super().__init__(repo, device, gen_params)
        self.context_length = 4096

    def prompt_template(self):
        return strip_all("""
        A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user"s questions.
        USER: 
        {prompt}
        ASSISTANT:""")

    def get_gen_params(self):
        return {
        "max_new_tokens": 512,
        "temperature": 0.7
    }

class GPT4ALL(Chatbot):

    def __init__(self, repo, device, gen_params=None) -> None:
        super().__init__(repo, device, gen_params)

    def get_gen_params(self):
        return {
        "max_new_tokens": 512,
        "temperature": 0.7
    }

class MPT(Chatbot):

    def __init__(self, repo, device, gen_params=None) -> None:
        super().__init__(repo, device, gen_params)

    def init_tokenizer(self):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    def get_model_params(self):
        config = AutoConfig.from_pretrained(self.repo[1], trust_remote_code=True)
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

    def __init__(self, repo, device, gen_params=None) -> None:
        super().__init__(repo, device, gen_params)
    
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

    def __init__(self, repo, device, gen_params=None) -> None:
        super().__init__(repo, device, gen_params)

    def prompt_template(self):
        return strip_all("""
        [INST] <<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don"t know the answer to a question, please don"t share false information.<</SYS>>{prompt}[/INST]""")
    
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

    def __init__(self, repo, device, gen_params=None) -> None:
        super().__init__(repo, device, gen_params)

    def prompt_template(self):
        return strip_all("""
        ### System: 
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

    def __init__(self, repo, device, gen_params=None) -> None:
        super().__init__(repo, device, gen_params)

    def prompt_template(self):
        return strip_all("""
        GPT4 User: {prompt}<|end_of_turn|>
        GPT4 Assistant:""")

    def get_gen_params(self):
        return {
                "max_new_tokens": 512,
                "temperature": 0.7
                }         
    
class BTLM(Chatbot):

    def __init__(self, repo, device, gen_params=None) -> None:
        super().__init__(repo, device, gen_params)

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

class Claude(Chatbot):

    def __init__(self, repo_id, gen_params=None) -> None:

        self.repo_id = (repo_id, get_repos()[repo_id])
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
        return ChatAnthropic(model=self.repo[1], **self.gen_params)
    
    def init_pipe(self):
        return self.model
import torch
import transformers
from transformers import AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, AutoConfig, AutoModelForCausalLM
from utils import the_bloke_repos
from enums import REPO_ID

def menu():

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
      break
      
  return repos[repo_id].value
  

def choose_bot(device, repo_id=None):

  if repo_id is None:
    repo_id = menu()

  device = f"cuda:{device}"

  if repo_id == "mosaicml/mpt-7b-chat":
  
    config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    #config.attn_config['attn_impl'] = 'triton'
    config.init_device = device 
    config.max_seq_len = 8192

    model = AutoModelForCausalLM.from_pretrained(
      repo_id,
      config=config,
      torch_dtype=torch.bfloat16, 
      trust_remote_code=True,
      #load_in_4bit=True
    )

    stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
    
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
    
    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    
    pipe = pipeline(model=model, tokenizer=tokenizer,
                    return_full_text=True,
                    task="text-generation",
                    device=device,
                    stopping_criteria=stopping_criteria,  
                    # temperature=0.7,  
                    # top_p=1,  
                    # top_k=0,  
                    max_new_tokens=512,  
                    repetition_penalty=1.1)
    
  elif repo_id == "tiiuae/falcon-7b-instruct":
  
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    model = AutoModelForCausalLM.from_pretrained(
      repo_id,
      torch_dtype=torch.bfloat16, 
      trust_remote_code=True,
      pad_token_id=tokenizer.eos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      num_return_sequences=1,
      do_sample=True,
      top_k=10,
      )
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=512,

    )
    
  elif repo_id == "TheBloke/GPT4All-13B-Snoozy-SuperHOT-8K-GPTQ":
  
    model_basename = "gpt4all-snoozy-13b-superhot-8k-GPTQ-4bit-128g.no-act.order"
    
    cfg = {
        "max_new_tokens": 512,
        #temperature: 0.7,
        #top_p: 0.95,
        #repetition_penalty: 1.15
    }
    
    pipe = the_bloke_repos(repo_id, model_basename, cfg)
  
  elif repo_id == "TheBloke/vicuna-7B-v1.3-GPTQ":
  
    model_basename = "vicuna-7b-v1.3-GPTQ-4bit-128g.no-act.order"
    
    cfg = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        #top_p: 0.95,
        #repetition_penalty: 1.15
    }
    
    pipe = the_bloke_repos(repo_id, model_basename, cfg)                
    
  elif repo_id == "TheBloke/Vicuna-13B-1-3-SuperHOT-8K-GPTQ":
  
    model_basename = "vicuna-13b-1.3.0-superhot-8k-GPTQ-4bit-128g.no-act.order"
    cfg = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        #top_p: 0.95,
        #repetition_penalty: 1.15
    }
    
    pipe = the_bloke_repos(repo_id, model_basename, cfg)
    
  elif repo_id == "TheBloke/Vicuna-33B-1-3-SuperHOT-8K-GPTQ":
  
    model_basename = "vicuna-33b-1.3-superhot-8k-GPTQ-4bit--1g.act.order"
    cfg = {
        "max_new_tokens": 512,
        #temperature: 0.7,
        #top_p: 0.95,
        #repetition_penalty: 1.15
    }
    
    pipe = the_bloke_repos(repo_id, model_basename, cfg)
    
  elif repo_id == "TheBloke/Llama-2-7b-Chat-GPTQ":
  
    model_basename = "gptq_model-4bit-128g"
    cfg = {
        "max_new_tokens": 512,
        #temperature: 0.7,
        #top_p: 0.95,
        #repetition_penalty: 1.15
    }
    
    pipe = the_bloke_repos(repo_id, model_basename, cfg)

  elif repo_id == "meta-llama/Llama-2-7b-chat-hf":

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    
    model = AutoModelForCausalLM.from_pretrained(
      repo_id,
      # torch_dtype=torch.bfloat16, 
      trust_remote_code=True,
      pad_token_id=tokenizer.eos_token_id,
      eos_token_id=tokenizer.eos_token_id,
      use_auth_token=True
      )
    
    pipe = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=512,
    )
    
  return pipe   
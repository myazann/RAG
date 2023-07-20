import argparse
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM

def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--device", default="1", type=str, choices=["0", "1"])
  
  args = parser.parse_args()
  
  return args
  
def available_repos():

  return {
    "1": "mosaicml/mpt-7b-chat",
    "3": "tiiuae/falcon-7b-instruct",
    "4": "TheBloke/GPT4All-13B-Snoozy-SuperHOT-8K-GPTQ",
    "5": "TheBloke/vicuna-7B-v1.3-GPTQ",
    "6": "TheBloke/Vicuna-13B-1-3-SuperHOT-8K-GPTQ" ,
    "7": "TheBloke/Vicuna-33B-1-3-SuperHOT-8K-GPTQ",
    "8": "TheBloke/Llama-2-7b-Chat-GPTQ"
  }
  
def the_bloke_repos(repo_id, model_basename, cfg):
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    
    model = AutoGPTQForCausalLM.from_quantized(repo_id,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=True,
            #device="cuda:0",
            use_triton=False,
            quantize_config=None)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **cfg)
    
    return pipe
import argparse
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM

def get_args():

   parser = argparse.ArgumentParser()
   parser.add_argument("-d", "--device", default="0", type=str, choices=["0", "1"])

   args = parser.parse_args()

   return args

def gptq_model_config():
   
   return {
      
      "use_safetensors": True,
      "trust_remote_code": True,
      "use_triton": False,
      "quantize_config": None
   }

def the_bloke_repos(repo_id, model_basename, cfg):
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    model_cfg = gptq_model_config()
    
    model = AutoGPTQForCausalLM.from_quantized(repo_id,
            model_basename=model_basename,
            **model_cfg)
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **cfg)
    
    return pipe
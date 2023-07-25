import argparse
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
import GPUtil

def get_args():

   parser = argparse.ArgumentParser()
   parser.add_argument("-d", "--device", default="0", type=str, choices=["0", "1", "cpu"])

   args = parser.parse_args()

   return args

def get_device():

   """
   Gets the GPU with the least memory usage. If there are no GPUs, sets device to CPU. 

   Note: GPUtil.getFirstAvailable gets the most available GPUs, but the definition 
   of availability depends on parameters such as maxMemory and maxLoad.
   For example, by default, it only gets the GPUs that have less than 50% memory usage.
   Check https://github.com/anderskm/gputil for more detailed info. 
   """
   
   print("GPU Usage:\n")
   GPUtil.showUtilization()

   devices = GPUtil.getAvailable()
   if len(devices) == 0:
      device = "cpu"
      print("There are either no GPUS, or they are too busy. Setting device to CPU!")
   else:
      deviceID = GPUtil.getFirstAvailable(order="memory")
      device = f"cuda:{deviceID[0]}"

   return device

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
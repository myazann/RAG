import argparse
from transformers import AutoTokenizer, pipeline
from auto_gptq import AutoGPTQForCausalLM
import GPUtil

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

def get_args():

   """

   parser = argparse.ArgumentParser()
   parser.add_argument("-d", "--device", default="0", type=str, choices=["0", "1", "cpu"])

   args = parser.parse_args()

   return args

   """
   pass

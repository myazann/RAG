import argparse
import GPUtil
import huggingface_hub
import os
import time

def init_env(project_name):

   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
   os.environ["LANGCHAIN_API_KEY"] = "ls__7eb356bde9434566bcbcac0b9ee5844b"

   timestamp = time.time()
   os.environ["LANGCHAIN_PROJECT"] = f"{project_name}_{timestamp}"

   huggingface_hub.login(new_session=False)

   device = get_device()
   print("Initialized huggingface hub and langsmith!")
   print("Returning command line arguments and device!")

   args = get_args()

   return args, device

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
      print(f"\nChosen GPU: {device}")
      device = f"cuda:{deviceID[0]}"

   return device

def get_args():

   parser = argparse.ArgumentParser()
   parser.add_argument("-doc", "--document", default="LESSEN_Project_Proposal.pdf", type=str)

   args = parser.parse_args()

   return args


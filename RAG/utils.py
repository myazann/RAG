import argparse
import os
import time
import difflib
from configparser import ConfigParser

import GPUtil
import huggingface_hub

def init_env(project_name):

   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
   os.environ["LANGCHAIN_API_KEY"] = "ls__7eb356bde9434566bcbcac0b9ee5844b"

   ls_name = f"{project_name}_{time.time()}"
   os.environ["LANGCHAIN_PROJECT"] = ls_name

   huggingface_hub.login(new_session=False)

   device = get_device()
   args = get_args()

   return args, device, ls_name

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
      print(f"\nChosen GPU: {deviceID}")
      device = f"cuda:{deviceID[0]}"

   return device

def get_cfg_params():

   parser = ConfigParser()
   parser.read("config.cfg")
   
   return parser

def get_args():

   parser = argparse.ArgumentParser()
   parser.add_argument("-doc", "--document", default="LESSEN_Project_Proposal.pdf", type=str)

   args = parser.parse_args()

   return args

def add_line_breaks(text, max_length):
   """
   If a line is too long, splits into shorter lines.
   Courtesy of Claude 2.0 :)
   """

   lines = text.split('\n')

   for i, line in enumerate(lines):
      if len(line) > max_length:
            
         words = line.split(' ')
         length = 0
         formatted = ""

         for word in words:
            if length + len(word) <= max_length:
               formatted += word + " "
               length += len(word) + 1
            else:
               formatted += "\n" + word + " "
               length = len(word) + 1
            
         lines[i] = formatted
      
   output = "\n".join(lines)
   return output

def strip_all(text):
   return "\n".join([line.strip() for line in text.splitlines()])

def find_best_substring_match(str1, str2):

    if len(str1) == len(str2):
        print("Strings have the same length, one string must be longer than the other!")
        return None
    
    if len(str1) > len(str2):
        db_str = str1
        query_str = str2
    else:
        db_str = str2
        query_str = str1
    
    n = len(query_str)
    best_ratio = 0
    best_match = ""
    for j in range(len(db_str)-n+1):
        substring = db_str[j:j+n]
        ratio = difflib.SequenceMatcher(None, query_str, substring).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = substring

    return best_ratio, best_match
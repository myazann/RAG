import argparse
import os
import random
from configparser import ConfigParser

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
      print(f"\nChosen GPU: {deviceID}")
      device = f"cuda:{deviceID[0]}"
   return device

def get_cfg_params():
   parser = ConfigParser()
   parser.read("config.cfg")
   
   return parser

def get_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("-doc", "--document", default=None, type=str)
   parser.add_argument("-ws", "--web_search", action=argparse.BooleanOptionalAction)
   args = parser.parse_args()

   return args

def add_line_breaks(text, max_length):
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

def list_files_in_directory(root_dir):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def shuffle_lists(list1, list2):
   zipped_list = list(zip(list1, list2))
   random.shuffle(zipped_list)
   list1_shuffled, list2_shuffled = zip(*zipped_list)
   list1_shuffled = list(list1_shuffled)
   list2_shuffled = list(list2_shuffled)
   return list1_shuffled, list2_shuffled
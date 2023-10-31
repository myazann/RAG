import argparse
import os
from configparser import ConfigParser

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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
   parser.add_argument("-doc", "--document", default="https://python.langchain.com/docs/get_started/introduction", type=str)
   parser.add_argument("-pt", "--perturb_test_type", default="test1", type=str)
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

def get_NoOpChain(llm):

   class NoOpLLMChain(LLMChain):

      def __init__(self):
            super().__init__(llm=llm, prompt=PromptTemplate(template="", input_variables=[]))

      def run(self, question: str, *args, **kwargs) -> str:
            return question

      async def arun(self, question: str, *args, **kwargs) -> str:
            return question
   
   return NoOpLLMChain()

def list_files_in_directory(root_dir):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list
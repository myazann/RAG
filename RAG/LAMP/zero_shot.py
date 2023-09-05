import json
import os
import urllib
import time

from langchain import LLMChain, PromptTemplate

from utils import get_device
from chatbots import choose_bot

with urllib.request.urlopen("https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/train/train_questions.json") as url:
    data = json.load(url)

with urllib.request.urlopen("https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/train/train_outputs.json") as url:
    gt_data = json.load(url)

device = get_device()

chatbot = choose_bot(device, model_name="CLAUDE-V2", gen_params={"max_new_tokens": 512, "temperature": 0})
test_name = f"LAMP_{chatbot.name}_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name

llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(chatbot.prompt_template))

for q in data:
    res = llm_chain.predict(prompt=q["input"])
    print(res)    
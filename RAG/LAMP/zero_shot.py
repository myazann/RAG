import json
import os
import urllib
import time

from langchain import LLMChain, PromptTemplate

from RAG.prompter import Prompter
from RAG.utils import get_device
from RAG.chatbots import choose_bot

with urllib.request.urlopen("https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/train/train_questions.json") as url:
    data = json.load(url)

with urllib.request.urlopen("https://ciir.cs.umass.edu/downloads/LaMP/LaMP_2/train/train_outputs.json") as url:
    gt_data = json.load(url)

device = get_device()
prompter = Prompter()
chatbot = choose_bot(device, model_name="CLAUDE-V2", gen_params={"max_new_tokens": 512, "temperature": 0})
lamp_prompt = prompter.merge_with_template(chatbot, "lamp")

test_name = f"LAMP_{chatbot.name}_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name

llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(lamp_prompt))

all_res = []
for i, q in enumerate(data):
    if i == 10:
        break
    article_idx = q["input"].rfind("article")
    res = llm_chain.predict(article=q["input"][(article_idx+len("article: ")):].strip()).strip()
    all_res.append(res)
    print(res)
    print(gt_data["golds"][i]["output"])


num_corr = sum([1 for i, pred in enumerate(all_res) if pred==gt_data["golds"][i]["output"]])
print(f"Accuracy: {num_corr/len(all_res)}")
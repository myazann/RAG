import os
import time

from langchain import LLMChain, PromptTemplate

from RAG.prompter import Prompter
from RAG.utils import get_device, get_args
from RAG.chatbots import choose_bot
from RAG.loader import FileLoader

device = get_device()
args = get_args()
dataset_num = args.lamp_dataset_num

data, gts = FileLoader.get_lamp_dataset(dataset_num)
prompter = Prompter()
chatbot = choose_bot(device, gen_params={"max_new_tokens": 256, "temperature": 0})
lamp_prompt = prompter.merge_with_template(chatbot, f"lamp_{dataset_num}")

test_name = f"LAMP_{chatbot.name}_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name

llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(lamp_prompt))

if dataset_num == "2":
    all_res = []
    for i, q in enumerate(data):
        if i == 10:
            break
        article_idx = q["input"].rfind("article")
        article = q["input"][(article_idx+len("article: ")):].strip()
        res = llm_chain.predict(article=article).strip()
        all_res.append(res)
        print(f"Article: {article}")
        print(f"Ground Truth: {gts[i]['output']}")
        print(f"Pred: {res}")

    num_corr = sum([1 for i, pred in enumerate(all_res) if pred==gts[i]["output"]])
    print(f"Accuracy: {num_corr/len(all_res)}")

elif dataset_num == "5":
    for i, q in enumerate(data):
        if i == 10:
            break
        abstract_idx = q["input"].find(":")+1
        abstract = q["input"][abstract_idx:].strip()
        res = llm_chain.predict(abstract=abstract).strip()
        print(f"Abstract: {abstract}")
        print(f"Ground Truth: {gts[i]['output']}")
        print(f"Pred: {res}")
import sys
import os
import time
import pickle

from langchain import LLMChain, PromptTemplate
from evaluate import load
import torch 

from RAG.prompter import Prompter
from RAG.utils import get_args
from RAG.chatbots import choose_bot
from RAG.loader import FileLoader

args = get_args()
dataset_num = args.lamp_dataset_num

data, gts = FileLoader.get_lamp_dataset(dataset_num)

prompter = Prompter()
chatbot = choose_bot(model_name="LLAMA2-7B-GGUF", gen_params={"max_new_tokens": 256}, q_bits=5)
lamp_prompt = prompter.merge_with_template(chatbot, f"lamp_{dataset_num}")

test_name = f"LAMP_{chatbot.name}_{chatbot.q_bit}-bit_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name
out_dir = "res_pkls"
os.makedirs(out_dir, exist_ok=True)
file_out_path = os.path.join(out_dir, f"LAMP_{dataset_num}_{chatbot.name}_{chatbot.q_bit}-bit.pkl")

llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(lamp_prompt))

if dataset_num == "5":
    all_res = []
    start_time = time.time()
    for i, q in enumerate(data):
        # print(f"Sample {i}:\n")
        abstract_idx = q["input"].find(":") + 1
        abstract = q["input"][abstract_idx:].strip()
        final_prompt = lamp_prompt.format(abstract=abstract)
        res = chatbot.pipe(final_prompt)
        all_res.append(res)
        # print(f"Abstract: \n{abstract}")
        # print(f"Ground Truth: \n{gts[i]}")
        # print(f"Pred: \n{res}")

        torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Took {(end_time-start_time)/3600} hours!")
    
with open(file_out_path, "wb") as f:
    pickle.dump(all_res, f)

"""
if dataset_num == "2":
    all_res = []
    for i, q in enumerate(data):
        if i == 10:
            break
        article_idx = q["input"].rfind("article")
        article = q["input"][(article_idx+len("article: ")):].strip()
        final_prompt = lamp_prompt.format(article=article)
        res = chatbot.pipe(final_prompt)
        all_res.append(res)
        print(f"Article: {article}")
        print(f"Ground Truth: {gts[i]['output']}")
        print(f"Pred: {res}")

    num_corr = sum([1 for i, pred in enumerate(all_res) if pred==gts[i]["output"]])
    print(f"Accuracy: {num_corr/len(all_res)}")

"""
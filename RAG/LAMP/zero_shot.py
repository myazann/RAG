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

orig_data, _ = FileLoader.get_lamp_dataset(dataset_num)

prompter = Prompter()
# ["LLAMA2-7B", "LLAMA2-7B-GGUF", "LLAMA2-13B", "LLAMA2-13B-GGUF", "VICUNA-7B-v1.5", "VICUNA-7B-v1.5-GGUF", "VICUNA-13B-v1.5", "VICUNA-13B-v1.5-GGUF"]
chatbot_names = ["VICUNA-7B-v1.5-GGUF"]
out_dir = "res_pkls"
os.makedirs(out_dir, exist_ok=True)

for chatbot_name in chatbot_names:

    print(chatbot_name)
    if "GGUF" in chatbot_name:
        q_bits = 5
        test_name = f"LAMP_{dataset_num}_{chatbot_name}_{q_bits}bit"
        file_out_path = os.path.join(out_dir, f"{test_name}.pkl")
    
    else:
        q_bits = None
        test_name = f"LAMP_{dataset_num}_{chatbot_name}"
        file_out_path = os.path.join(out_dir, f"{test_name}.pkl")

    all_res = []
    if os.path.exists(file_out_path):
        with open(file_out_path, "rb") as f:
             all_res = pickle.load(f)

        if len(all_res) == len(orig_data):
            print("Experiment for this chatbot is already concluded!")
            continue

        else:
            data = orig_data[len(all_res):]
            
    os.environ["LANGCHAIN_PROJECT"] = test_name
    chatbot = choose_bot(model_name=chatbot_name, gen_params={"max_new_tokens": 64}, q_bits=q_bits)
    lamp_prompt = prompter.merge_with_template(chatbot, f"lamp_{dataset_num}")

    llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(lamp_prompt))

    print(f"Starting from sample no. {len(all_res)}")
    if dataset_num == "5":
        
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
            with open(file_out_path, "wb") as f:
                pickle.dump(all_res, f)

        end_time = time.time()
        print(f"Took {(end_time-start_time)/3600} hours!")

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
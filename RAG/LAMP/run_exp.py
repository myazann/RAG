import os
import time
import pickle
import sys
import torch 
import subprocess

from RAG.prompter import Prompter
from RAG.chatbots import choose_bot
from RAG.utils import shuffle_lists
from lamp_utils import get_lamp_args, create_retr_data, retrieved_idx, get_lamp_dataset, get_profvar_names

args = get_lamp_args()
q_type = args.quant
dataset_num = args.dataset_num
dataset_split = args.dataset_split
k = args.k
retriever = args.retriever if k != 0 else None
max_context_length = args.max_context_length

FINAL_DB_SIZE = {
    3: {
        "train_dev": 22388,
        "dev": 2487
    },
    5: {
        "train_dev": 12121,
        "dev": 2487
    }  
}
MAX_NEW_TOKENS = 64

data, out_gts = get_lamp_dataset(dataset_num)
prof_text_name, prof_gt_name, prof_prompt_name = get_profvar_names(dataset_num)
prompter = Prompter()
chatbot_names = ["LLAMA3-8B", "LLAMA2-7B", "MISTRAL-7B-v0.1-INSTRUCT", "ZEPHYR-7B-BETA", "STARLING-7B-ALPHA", "OPENCHAT-3.5"]
if k == "0":
    out_dir = f"res_pkls/D{dataset_num}/{dataset_split}/K{k}"
else:
    out_dir = f"res_pkls/D{dataset_num}/{dataset_split}/K{k}/{retriever}"
os.makedirs(out_dir, exist_ok=True)
print(f"Running experiments for the {dataset_num}th dataset with k={k} and {retriever}")
for chatbot_name in chatbot_names:
    if chatbot_name in ["LLAMA2-70B", "YI-34B-CHAT", "MISTRAL-8x7B-v0.1-INSTRUCT"]:
        if q_type is None:
            print("This model cannot be run unquantized!") 
            continue
        elif q_type == "GPTQ":
            print("GPTQ implementation of this model is not stable!")
            continue
    if q_type is not None:
        chatbot = choose_bot(model_name=f"{chatbot_name}-{q_type}", gen_params={"max_new_tokens": MAX_NEW_TOKENS})
    else:
        chatbot = choose_bot(model_name=chatbot_name, gen_params={"max_new_tokens": MAX_NEW_TOKENS})
    if "max" in k and int(chatbot.context_length) > int(max_context_length):
        exp_window = int(int(max_context_length)/1000)
        chatbot_name = f"{chatbot_name}-{exp_window}K"
        chatbot.context_length = max_context_length
    if q_type is not None:
        chatbot_name = f"{chatbot_name}-{q_type}"
    print(subprocess.run("gpustat"))
    print(chatbot_name)
    file_out_path = f"{out_dir}/{chatbot_name}.pkl"
    if os.path.exists(file_out_path):
        with open(file_out_path, "rb") as f:
             all_res = pickle.load(f)
    else:
        all_res = []
    if len(all_res) == FINAL_DB_SIZE[dataset_num][dataset_split]:
        print("Experiment for this chatbot is already concluded!")
        continue
    else:
        orig_queries, orig_prof_texts, orig_prof_gts, _, _ = create_retr_data(data[dataset_split], out_gts[dataset_split], dataset_num)
        queries = orig_queries[len(all_res):]
        prof_texts = orig_prof_texts[len(all_res):]
        prof_gts = orig_prof_gts[len(all_res):]
    if k != "0":
        retr_doc_idxs = retrieved_idx(prof_texts, queries, dataset_num, dataset_split, retriever)
        retr_doc_idxs = retr_doc_idxs[len(all_res):]
    print(f"Starting from sample no. {len(all_res)}")
    start_time = time.time()
    sys.stdout.flush()
    skip_k = 0
    doc_k = k
    if "_" in k:
        doc_k = k.split("_")[0]
        if "skip" in k:
            skip_k = int(k[k.find("skip_")+len("skip_")])
    for i in range(len(queries)):
        if k == "0":
            lamp_prompt = prompter.lamp_prompt(dataset_num, prof_text=queries[i])     
        else:
            retr_docs = retr_doc_idxs[i]
            example_pairs = ""
            if "max" in k:
                doc_k = len(retr_docs)-skip_k
            else:
                doc_k = int(doc_k)
            retr_texts = [prof_texts[i][doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
            retr_gts = [prof_gts[i][doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
            if k.endswith("shuffle"):
                retr_texts, retr_gts = shuffle_lists(retr_texts, retr_gts)
            if k.endswith("reverse"):
                retr_texts = retr_texts[::-1]
                retr_gts = retr_gts[::-1]
            for text, gt in zip(retr_texts, retr_gts):
                example = f"""{prof_prompt_name.capitalize()}:\n{text}\n{prof_gt_name.capitalize()}:\n{gt}\n"""  
                lamp_prompt = prompter.lamp_prompt(dataset_num, prof_text=queries[i], examples=example_pairs)
                avail_space = int(chatbot.context_length) - chatbot.count_tokens(lamp_prompt)
                if chatbot.count_tokens(example_pairs + "\n" + example + queries[i]) < avail_space:
                    example_pairs = example_pairs + "\n" + example   
                else:
                    break   
            lamp_prompt = prompter.lamp_prompt(dataset_num, prof_text=queries[i], examples=example_pairs)
        res = chatbot.prompt_chatbot(lamp_prompt)
        all_res.append(res)
        if (i+1)%500==0 or (i+1)==len(queries):
            print(i)
            with open(file_out_path, "wb") as f:
                pickle.dump(all_res, f)
        sys.stdout.flush()
    end_time = time.time()
    print(f"Took {(end_time-start_time)/3600} hours!")
    del chatbot
    chatbot = []
    torch.cuda.empty_cache()
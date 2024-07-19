import os
import time
import json
import sys
import torch 
import subprocess

from RAG.prompter import Prompter
from RAG.chatbot import choose_bot
from RAG.utils import shuffle_lists
from RAG.output_formatter import lamp_output_formatter
from lamp_utils import get_lamp_args, create_retr_data, retrieved_idx, get_lamp_dataset, get_profvar_names, log_exp

args = get_lamp_args()
q_type = args.quant
dataset_num = args.dataset_num
dataset_split = args.dataset_split
k = args.k
retriever = args.retriever if k != 0 else None
MAX_NEW_TOKENS = 64

data, _ = get_lamp_dataset(dataset_num, dataset_split)
prof_text_name, prof_gt_name, prof_prompt_name = get_profvar_names(dataset_num)
prompter = Prompter()
LLMs = ["LLAMA3-8B", "LLAMA3-70B", "GEMMA-2-9B", "GEMMA-2-27B"]
if k == "0":
    out_dir = f"res_pkls/D{dataset_num}/{dataset_split}/K{k}"
else:
    out_dir = f"res_pkls/D{dataset_num}/{dataset_split}/K{k}/{retriever}"
os.makedirs(out_dir, exist_ok=True)
print(f"Running experiments for the {dataset_num}th dataset with k={k} and {retriever} on {dataset_split} set")
for model_name in LLMs:
    if q_type is not None:
        model_name = f"{model_name}-{q_type}"
    file_out_path = f"{out_dir}/{model_name}.json"
    if os.path.exists(file_out_path):
        with open(file_out_path, "rb") as f:
             all_res = json.load(f)["golds"]
    else:
        all_res = []
    print(model_name)   
    if len(all_res) == len(data):
        print("Experiment for this llm is already concluded!")
        continue
    else:
        llm = choose_bot(model_name=model_name, gen_params={"max_new_tokens": MAX_NEW_TOKENS})
        print(subprocess.run("gpustat"))    
        exp_name = f"{dataset_split}_{dataset_num}_{model_name}_K{k}_{retriever}"
        orig_queries, orig_prof_texts, orig_prof_gts = create_retr_data(data, dataset_num)
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
                avail_space = int(llm.context_length) - llm.count_tokens(lamp_prompt)
                if llm.count_tokens(example_pairs + "\n" + example + queries[i]) < avail_space:
                    example_pairs = example_pairs + "\n" + example   
                else:
                    break   
            lamp_prompt = prompter.lamp_prompt(dataset_num, prof_text=queries[i], examples=example_pairs)
        start_bot_time = time.time()    
        res = llm.prompt_chatbot(lamp_prompt)
        end_bot_time = time.time()
        formatted_res = lamp_output_formatter(res, dataset_num)
        all_res.append({
            "id": data[i]["id"],
            "output": formatted_res
        })
        cur_iter_res = {
            "prompt": lamp_prompt,
            "output": res,
            "formatted_output": formatted_res,
            "model_inf_time": round(end_bot_time - start_bot_time, 2) 
        }
        log_exp(cur_iter_res, exp_name)
        if (i+1)%500==0 or (i+1)==len(queries):
            print(i)
            with open(file_out_path, "w") as f:
                json.dump({
                    "task": f"LaMP_{dataset_num}",
                    "golds": all_res
                }, f)
        sys.stdout.flush()
    end_time = time.time()
    print(f"Took {(end_time-start_time)/3600} hours!")
    del llm
    llm = []
    torch.cuda.empty_cache()
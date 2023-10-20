import os
import time
import pickle
import sys

from langchain import LLMChain, PromptTemplate
from evaluate import load
import torch 

from RAG.prompter import Prompter
from RAG.chatbots import choose_bot, get_model_cfg
from RAG.loader import FileLoader
from lamp_utils import get_lamp_args, create_retr_data, retrieved_idx

args = get_lamp_args()
is_q = bool(args.quant_bots)
dataset_num = args.dataset_num
k = args.k
retriever = args.retriever if k != 0 else None
context_length = args.context_length

Q_BIT = 5 if is_q else None
FINAL_DB_SIZE = 12121
MAX_NEW_TOKENS = 64

data, out_gts = FileLoader.get_lamp_dataset(dataset_num)
prompter = Prompter()
chatbot_names = ["LLAMA2-7B", "LLAMA2-13B", "VICUNA-7B-v1.5", "VICUNA-13B-v1.5", "MISTRAL-7B-v0.1-INSTRUCT"]
if is_q:
    chatbot_names = [f"{bot_name}-GGUF" for bot_name in chatbot_names]
if k == "0":
    out_dir = f"res_pkls/D{dataset_num}/K{k}"
else:
    out_dir = f"res_pkls/D{dataset_num}/K{k}/{retriever}"
os.makedirs(out_dir, exist_ok=True)
print(f"Running experiments for the {dataset_num}th dataset with k={k} with {retriever}")
for chatbot_name in chatbot_names:
    print(chatbot_name)
    if k == "0":
        test_name = f"LAMP_D{dataset_num}_K{k}_{chatbot_name}"   
    else:
        test_name = f"LAMP_D{dataset_num}_K{k}_{retriever}_{chatbot_name}"
    file_out_path = os.path.join(out_dir, f"{chatbot_name}")
    def_ctx_length = get_model_cfg()[chatbot_name]["context_length"]
    if def_ctx_length != context_length:
        exp_window = int(int(context_length)/1000)
        test_name = f"{test_name}_{exp_window}K"
        file_out_path = f"{file_out_path}_{exp_window}K"
    os.environ["LANGCHAIN_PROJECT"] = test_name
    file_out_path = f"{file_out_path}.pkl"
    if os.path.exists(file_out_path):
        with open(file_out_path, "rb") as f:
             all_res = pickle.load(f)
    else:
        all_res = []
    if len(all_res) == FINAL_DB_SIZE:
        print("Experiment for this chatbot is already concluded!")
        continue
    else:
        orig_queries, orig_corpuses, orig_titles, _ = create_retr_data(data["train_dev"], out_gts["train_dev"])
        queries = orig_queries[len(all_res):]
        corpuses = orig_corpuses[len(all_res):]
        titles = orig_titles[len(all_res):]
    chatbot = choose_bot(model_name=chatbot_name, gen_params={"max_new_tokens": MAX_NEW_TOKENS}, q_bits=Q_BIT)
    chatbot.context_length = context_length
    if k == "0":
        lamp_prompt = prompter.merge_with_template(chatbot, f"lamp_{dataset_num}")
    else:
        lamp_prompt = prompter.merge_with_template(chatbot, f"lamp_{dataset_num}_with_k")
        retr_doc_idxs = retrieved_idx(corpuses, queries, retriever)
        retr_doc_idxs = retr_doc_idxs[len(all_res):]
    llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(lamp_prompt))
    print(f"Starting from sample no. {len(all_res)}")
    start_time = time.time()
    avail_space = int(chatbot.context_length) - chatbot.count_tokens(lamp_prompt)
    sys.stdout.flush()
    skip_k = 0
    doc_k = k
    if "skip" in k:
        doc_k = k.split("_")[0]
        skip_k = int(k.split("_")[-1])
    for i in range(len(queries)):
        if k == "0":
            final_prompt = lamp_prompt.format(abstract=queries[i])        
        else:
            retr_docs = retr_doc_idxs[i]
            example_pairs = ""
            if k == "max":
                doc_k = len(retr_docs)-skip_k
            else:
                doc_k = int(doc_k)
            retr_corpuses = [corpuses[i][doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
            retr_titles = [titles[i][doc_id] for doc_id in retr_docs[skip_k: (doc_k+skip_k)]]
            for corp, title in zip(retr_corpuses, retr_titles):
                example = f"""Abstract:\n{corp}\nTitle:\n{title}\n"""  
                if chatbot.count_tokens(example_pairs + "\n" + example + queries[i]) < avail_space:
                    example_pairs = example_pairs + "\n" + example   
                else:
                    break   
            final_prompt = lamp_prompt.format(examples=example_pairs, abstract=queries[i])
        res = chatbot.pipe(final_prompt)
        all_res.append(res)
        torch.cuda.empty_cache()
        if (i+1)%500==0 or (i+1)==len(queries):
            with open(file_out_path, "wb") as f:
                pickle.dump(all_res, f)
    end_time = time.time()
    print(f"Took {(end_time-start_time)/3600} hours!")
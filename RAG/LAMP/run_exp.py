import os
import time
import pickle
import sys

from langchain import LLMChain, PromptTemplate
from evaluate import load
import torch 

from RAG.prompter import Prompter
from RAG.chatbots import choose_bot
from lamp_utils import get_lamp_args, create_retr_data, retrieved_idx, get_lamp_dataset

args = get_lamp_args()
q_type = args.quant
q_bits = args.q_bits
dataset_num = args.dataset_num
k = args.k
retriever = args.retriever if k != 0 else None
max_context_length = args.max_context_length

FINAL_DB_SIZE = 12121
MAX_NEW_TOKENS = 64

data, out_gts = get_lamp_dataset(dataset_num)
prompter = Prompter()
chatbot_names = ["LLAMA2-7B", "LLAMA2-13B", "LLAMA2-70B", "VICUNA-7B-16K-v1.5", "VICUNA-13B-16K-v1.5", "MISTRAL-7B-v0.1-INSTRUCT", "ZEPHYR-7B-ALPHA", "ZEPHYR-7B-BETA"]
# chatbot_names = ["VICUNA-7B-v1.5", "VICUNA-13B-v1.5"]
if q_type is not None:
    chatbot_names = [f"{bot_name}-{q_type}" for bot_name in chatbot_names]
if k == "0":
    out_dir = f"res_pkls/D{dataset_num}/K{k}"
else:
    out_dir = f"res_pkls/D{dataset_num}/K{k}/{retriever}"
os.makedirs(out_dir, exist_ok=True)
print(f"Running experiments for the {dataset_num}th dataset with k={k} and {retriever}")
for chatbot_name in chatbot_names:
    if "LLAMA2-70B" in chatbot_name:
        if q_type is None:
            print("Unquantized LLaMA2-70B cannot be run!") 
            continue
        elif q_type == "GGUF" and int(q_bits) > 4:
            print("LLaMA2-70B can only be run in 4-bits (or less) with GGUF quantization!")
            continue
    if k == "0":
        test_name = f"LAMP_D{dataset_num}_K{k}"   
    else:
        test_name = f"LAMP_D{dataset_num}_K{k}_{retriever}"
    #file_out_path = os.path.join(out_dir, f"{bit_chatbot_name}")
    chatbot = choose_bot(model_name=chatbot_name, gen_params={"max_new_tokens": MAX_NEW_TOKENS}, q_bits=q_bits)
    if k == "max" and int(chatbot.context_length) > int(max_context_length):
        exp_window = int(int(max_context_length)/1000)
        # test_name = f"{test_name}-{exp_window}K"
        chatbot_name = f"{chatbot_name}-{exp_window}K"
        chatbot.context_length = max_context_length
    if q_type == "GGUF":
        chatbot_name = f"{chatbot_name}-{q_bits}_bits"
    print(chatbot_name)
    os.environ["LANGCHAIN_PROJECT"] = f"{test_name}_{chatbot_name}"
    file_out_path = f"{out_dir}/{chatbot_name}.pkl"
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
            if "max" in k:
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
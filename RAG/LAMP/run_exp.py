import os
import time
import pickle
import subprocess
import sys

from langchain import LLMChain, PromptTemplate
from evaluate import load
import torch 

from RAG.prompter import Prompter
from RAG.utils import get_args
from RAG.chatbots import choose_bot
from RAG.loader import FileLoader
from lamp_retrievers import create_retr_data
    
args = get_args()
dataset_num = args.lamp_dataset_num
k = args.lamp_k

data, out_gts = FileLoader.get_lamp_dataset(dataset_num)
prompter = Prompter()
# chatbot_names = ["LLAMA2-7B", "LLAMA2-7B-GGUF", "LLAMA2-13B", "LLAMA2-13B-GGUF", "VICUNA-7B-16K-v1.5", "VICUNA-7B-16K-v1.5-GGUF", "VICUNA-13B-16K-v1.5", "VICUNA-13B-16K-v1.5-GGUF"]
chatbot_names = ["LLAMA2-7B", "LLAMA2-13B", "VICUNA-7B-16K-v1.5", "VICUNA-13B-16K-v1.5"]
out_dir = f"res_pkls/D{dataset_num}/K{k}"
os.makedirs(out_dir, exist_ok=True)

print(f"Running experiments for the {dataset_num}th dataset with k={k}")
for chatbot_name in chatbot_names:

    print(chatbot_name)
    if "GGUF" in chatbot_name:
        q_bits = 5
        test_name = f"LAMP_D{dataset_num}_K{k}_{chatbot_name}_{q_bits}bit"    
    else:
        q_bits = None
        test_name = f"LAMP_D{dataset_num}_K{k}_{chatbot_name}"
    os.environ["LANGCHAIN_PROJECT"] = test_name
    file_out_path = os.path.join(out_dir, f"{chatbot_name}.pkl")

    if os.path.exists(file_out_path):
        with open(file_out_path, "rb") as f:
             all_res = pickle.load(f)
    else:
        all_res = []

    if len(all_res) == 12121:
        print("Experiment for this chatbot is already concluded!")
        continue

    else:
        orig_queries, orig_corpuses, orig_titles, out_gts = create_retr_data(data, out_gts)
        queries = orig_queries[len(all_res):]
        corpuses = orig_corpuses[len(all_res):]
        titles = orig_titles[len(all_res):]
    
    chatbot = choose_bot(model_name=chatbot_name, gen_params={"max_new_tokens": 64}, q_bits=q_bits)
    print(subprocess.run("nvidia-smi"))
    
    if k == "0":
        lamp_prompt = prompter.merge_with_template(chatbot, f"lamp_{dataset_num}")
    else:
        lamp_prompt = prompter.merge_with_template(chatbot, f"lamp_{dataset_num}_with_k")

    llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(lamp_prompt))

    print(f"Starting from sample no. {len(all_res)}")
    sys.stdout.flush()
    start_time = time.time()
    for i, q in enumerate(queries):
        if k == "0":
            final_prompt = lamp_prompt.format(abstract=queries[i])        
        res = chatbot.pipe(final_prompt)
        all_res.append(res)

        torch.cuda.empty_cache()
        if (i+1)%500==0 or (i+1)==len(queries):
            with open(file_out_path, "wb") as f:
                pickle.dump(all_res, f)

    end_time = time.time()
    print(f"Took {(end_time-start_time)/3600} hours!")
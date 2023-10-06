import os
import time
import pickle
import sys

from langchain import LLMChain, PromptTemplate
from evaluate import load
import torch 

from RAG.prompter import Prompter
from RAG.chatbots import choose_bot
from RAG.loader import FileLoader
from lamp_utils import get_lamp_args, create_retr_data, retrieved_idx

args = get_lamp_args()
dataset_num = args.dataset_num
k = args.k
retriever = args.retriever

data, out_gts = FileLoader.get_lamp_dataset(dataset_num)
prompter = Prompter()
# chatbot_names = ["LLAMA2-7B", "LLAMA2-7B-GGUF", "LLAMA2-13B", "LLAMA2-13B-GGUF", "VICUNA-7B-16K-v1.5", "VICUNA-7B-16K-v1.5-GGUF", "VICUNA-13B-16K-v1.5", "VICUNA-13B-16K-v1.5-GGUF"]
chatbot_names = ["LLAMA2-7B", "LLAMA2-13B", "VICUNA-7B-v1.5", "VICUNA-13B-v1.5"]
if k == "0":
    out_dir = f"res_pkls/D{dataset_num}/K{k}"
else:
    out_dir = f"res_pkls/D{dataset_num}/K{k}/{retriever}"
os.makedirs(out_dir, exist_ok=True)

print(f"Running experiments for the {dataset_num}th dataset with k={k} with {retriever}")
for chatbot_name in chatbot_names:

    print(chatbot_name)
    if "GGUF" in chatbot_name:
        q_bits = 5
    else:
        q_bits = None
    if k == "0":
        test_name = f"LAMP_D{dataset_num}_K{k}_{chatbot_name}"   
    else:
        test_name = f"LAMP_D{dataset_num}_K{k}_{retriever}_{chatbot_name}"
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
    for i, q in enumerate(queries):
        if k == "0":
            final_prompt = lamp_prompt.format(abstract=queries[i])        
        else:
            example_pairs = ""
            if k == "max":
                i_retr = 0
                while i_retr < len(retr_doc_idxs[i]):
                    example = f"""Abstract:\n{corpuses[i][i_retr]}\nTitle:\n{titles[i][i_retr]}\n""" 
                    if chatbot.count_tokens(example_pairs + "\n" + example + queries[i]) < avail_space:
                        example_pairs = example_pairs + "\n" + example
                        i_retr += 1
                    else:
                        break                
            else:
                retr_corpuses = [corpuses[i][doc_id] for doc_id in retr_doc_idxs[i][:int(k)]]
                retr_titles = [titles[i][doc_id] for doc_id in retr_doc_idxs[i][:int(k)]]
                for corp, title in zip(retr_corpuses, retr_titles):
                    example_pairs = example_pairs + f"""Abstract:\n{corp}\nTitle:\n{title}\n"""      

            final_prompt = lamp_prompt.format(examples=example_pairs, abstract=queries[i])
        res = chatbot.pipe(final_prompt)
        all_res.append(res)

        torch.cuda.empty_cache()
        if (i+1)%500==0 or (i+1)==len(queries):
            with open(file_out_path, "wb") as f:
                pickle.dump(all_res, f)

    end_time = time.time()
    print(f"Took {(end_time-start_time)/3600} hours!")
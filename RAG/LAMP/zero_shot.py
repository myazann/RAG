import sys
import os
import time
import subprocess

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

chatbot = choose_bot(model_name="LLAMA2-70B-GGUF", model_params={"n_gpu_layers": 0,
                "n_batch": 512,
                "verbose": True,
                "n_ctx": 4096}, gen_params={"max_new_tokens": 256}, q_bits=5)
print(subprocess.run("nvidia-smi"))

sys.stdout.flush()
lamp_prompt = prompter.merge_with_template(chatbot, f"lamp_{dataset_num}")

test_name = f"LAMP_{chatbot.name}_{chatbot.q_bit}-bit_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name

llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(lamp_prompt))

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

elif dataset_num == "5":
    for i, q in enumerate(data):
        if i == 10:
            break
        print(f"Sample {i}:\n")
        abstract_idx = q["input"].find(":") + 1
        abstract = q["input"][abstract_idx:].strip()
        print(subprocess.run("gpustat"))
        final_prompt = lamp_prompt.format(abstract=abstract)
        res = chatbot.pipe(final_prompt)
        gt = gts[i]['output']
        print(f"Abstract: \n{abstract}")
        print(f"Ground Truth: \n{gt}")
        print(f"Pred: \n{res}")

        """
        bertscore = load("bertscore")
        bertscore_res = bertscore.compute(predictions=[res], references=[gt], lang="en")
        print(f"Bertscore: {bertscore_res}")

        rouge = load("rouge")
        rouge_results = rouge.compute(predictions=[res], references=[gt])
        print(f"Rouge: {rouge_results}")

        bleu = load("bleu")
        bleu_results = bleu.compute(predictions=[res], references=[gt])
        print(f"BLEU: {bleu_results}")
        """
        torch.cuda.empty_cache()
        sys.stdout.flush()
import os
import pickle

import pandas as pd
from evaluate import load

from RAG.output_formatter import lamp_output_formatter
from RAG.loader import FileLoader
from lamp_utils import get_lamp_args, create_retr_data

args = get_lamp_args()
dataset_num = args.dataset_num
k = args.k
retriever = args.retriever
if k == "0":
    out_dir = os.path.join("res_pkls", f"D{dataset_num}", f"K{k}")
else:
    out_dir = os.path.join("res_pkls", f"D{dataset_num}", f"K{k}", retriever)

data, out_gts = FileLoader.get_lamp_dataset(dataset_num)
_, _, _, out_gts = create_retr_data(data, out_gts)

all_res_files = sorted(os.listdir(out_dir))
if k == "0":
    print("Running eval for zero-shot!")
else:
    print(f"Running eval for K={k} and {retriever}")

all_rouge = []
cols = []
for file in all_res_files:
    with open(os.path.join(out_dir, file), "rb") as f:
        all_res = pickle.load(f)

    if len(all_res) != len(out_gts):
        continue
    cols.append(file[:-4])
    all_res = [lamp_output_formatter(res) for res in all_res]

    rouge = load("rouge")
    rouge_results = rouge.compute(predictions=all_res, references=out_gts)
    all_rouge.append(rouge_results)

df = pd.DataFrame(all_rouge)
df.index = cols
print(df.sort_values("rougeLsum", ascending=False))

"""

    bertscore = load("bertscore")
    bertscore_res = bertscore.compute(predictions=all_res, references=out_gts, lang="en", device="cuda:0")
    print(f"Bertscore: {bertscore_res} \n")
    all_rouge_list = []
    all_bleu_list = []
    i = 0
    for res, gt in zip(all_res, out_gts):
        print(i)
        rouge_results = rouge.compute(predictions=[res], references=[gt])
        all_rouge_list.append(rouge_results)
        bleu_results = bleu.compute(predictions=[res], references=[gt])
        all_bleu_list.append(bleu_results)
        i += 1

    all_rouge_dict = {}
    rouge_keys = all_rouge_list[0].keys()
    for key in rouge_keys:
        all_rouge_dict[key] = [rouge_res[key] for rouge_res in all_rouge_list]

    all_bleu_dict = {}
    bleu_keys = all_bleu_list[0].keys()
    for key in bleu_keys:
        all_bleu_dict[key] = [rouge_res[key] for rouge_res in all_bleu_list]
   
    print(f"Rouge: {all_rouge_dict} \n")
    print(f"BLEU: {all_bleu_dict}")
"""
import os
import pickle

from evaluate import load

from RAG.utils import get_args
from RAG.loader import FileLoader

out_dir = "res_pkls"

args = get_args()
dataset_num = args.lamp_dataset_num
_, gts = FileLoader.get_lamp_dataset(dataset_num, get_gts_only=True)

all_res_files = [file for file in os.listdir(out_dir) if file.split("_")[1] == dataset_num]

for file in all_res_files:
    with open(os.path.join(out_dir, file), "rb") as f:
        all_res = pickle.load(f)

    if len(all_res) != len(gts):
        gts = gts[:len(all_res)]
    bertscore = load("bertscore")
    bertscore_res = bertscore.compute(predictions=all_res, references=gts, lang="en")
    print(f"Bertscore: {bertscore_res} \n")

    rouge = load("rouge")
    bleu = load("bleu")

    all_rouge_list = []
    all_bleu_list = []
    for res, gt in zip(all_res, gts):
        rouge_results = rouge.compute(predictions=[res], references=[gt])
        all_rouge_list.append(rouge_results)
        bleu_results = bleu.compute(predictions=[res], references=[gt])
        all_bleu_list.append(bleu_results)

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
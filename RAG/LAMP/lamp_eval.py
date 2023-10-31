import os
import pickle

import pandas as pd
from evaluate import load

from RAG.output_formatter import lamp_output_formatter
from lamp_utils import get_lamp_args, create_retr_data, get_val_idx, get_lamp_dataset

def list_files_in_directory(root_dir):
    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

all_res_files = sorted(list_files_in_directory("res_pkls"))
args = get_lamp_args()
dataset_num = args.dataset_num
data, out_gts = get_lamp_dataset(dataset_num)
_, _, _, out_gts = create_retr_data(data["train_dev"], out_gts["train_dev"])
val_idx = get_val_idx(out_gts)
all_rouge = []
models = []
for file in all_res_files:
    with open(file, "rb") as f:
        preds = pickle.load(f)
    if len(preds) != len(out_gts):
        continue
    params = file.split("/")
    if len(params) == 4:
        k = "0"
        retriever = None
    else:
        k = params[-3][1:] 
        retriever = file.split("/")[-2]
    model_name = file.split("/")[-1][:-4]
    models.append(model_name)
    print(k, retriever, model_name)
    preds = [preds[i] for i in val_idx]
    gts = [out_gts[i] for i in val_idx]
    preds = [lamp_output_formatter(res) for res in preds]
    rouge = load("rouge")
    rouge_results = rouge.compute(predictions=preds, references=gts)
    rouge_results["k"] = k
    rouge_results["retriever"] = retriever
    all_rouge.append(rouge_results)
df = pd.DataFrame(all_rouge)
df["model"] = models
df = df[["model", "retriever", "k", "rouge1", "rouge2", "rougeL", "rougeLsum"]]
df = df.round(dict([(c, 4) for c in df.columns if "rouge" in c]))
df.sort_values("rougeLsum", ascending=False).to_csv("lamp_eval_res.csv", index=False)
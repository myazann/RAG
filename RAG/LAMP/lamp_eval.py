import pickle
import pandas as pd
from evaluate import load
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

from RAG.utils import list_files_in_directory
from RAG.output_formatter import lamp_output_formatter
from lamp_utils import get_lamp_args, create_retr_data, get_lamp_dataset

args = get_lamp_args()
dataset_num = args.dataset_num
dataset_split = args.dataset_split
all_res_files = sorted(list_files_in_directory(f"res_pkls/D{dataset_num}/{dataset_split}"))
data, out_gts = get_lamp_dataset(dataset_num)
_, _, _, out_gts, _ = create_retr_data(data[dataset_split], out_gts[dataset_split], dataset_num)
all_res = []
models = []
cols = ["model", "retriever", "k"]
if dataset_num > 3:
    rouge = load("rouge")
    cols.extend(["rouge1", "rouge2", "rougeL", "rougeLsum"])
else:
    cols.extend(["acc", "f1_macro", "mae", "rmse"])
for file in all_res_files:
    with open(file, "rb") as f:
        preds = pickle.load(f)
    if len(preds) != len(out_gts):
        continue
    params = file.split("/")
    if len(params) == 5:
        k = "0"
        retriever = None
    else:
        k = params[-3][1:] 
        retriever = file.split("/")[-2]
    model_name = file.split("/")[-1][:-4]
    models.append(model_name)
    print(k, retriever, model_name)
    preds = [lamp_output_formatter(pred, dataset_num) for pred in preds]
    if dataset_num > 3:
        rouge_results = rouge.compute(predictions=preds, references=out_gts)
        rouge_results["k"] = k
        rouge_results["retriever"] = retriever
        all_res.append(rouge_results)
    else:
        f1_macro = f1_score(out_gts, preds, average="macro")
        mae = mean_absolute_error(list(map(int, out_gts)), list(map(int, preds)))
        rmse = mean_squared_error(list(map(int, out_gts)), list(map(int, preds)))
        print(f"F1 Macro: {f1_macro}")
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        cor_pred = 0
        for i in range(len(out_gts)):
            if str(out_gts[i]) == str(preds[i]):
                cor_pred += 1
        acc = cor_pred/len(out_gts)
        print(f"Accuracy: {acc}")
        print()
        all_res.append({
            "k": k,
            "retriever": retriever,
            "acc": acc,
            "f1_macro": f1_macro,
            "mae": mae,
            "rmse": rmse
        })
df = pd.DataFrame(all_res)
df["model"] = models
df = df[cols]
df = df.round(dict([(c, 4) for c in df.columns if df[c].dtype == "float64"]))
df.to_csv(f"lamp_{dataset_num}_eval_res.csv", index=False)
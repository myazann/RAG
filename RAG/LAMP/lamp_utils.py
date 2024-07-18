import argparse
from itertools import chain
import os
import pickle
import json
import urllib

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from contriever.src.contriever import Contriever

def get_lamp_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("-q", "--quant", default=None, type=str)
   parser.add_argument("-dn", "--dataset_num", default=5, type=int)
   parser.add_argument("-ds", "--dataset_split", default="train_dev", type=str)
   parser.add_argument("-k", "--k", default="3", type=str)
   parser.add_argument("-r", "--retriever", default="bm25", type=str)
   parser.add_argument("-mcl", "--max_context_length", default=4096, type=int)
   return parser.parse_args()

def get_lamp_dataset(dataset_num, mode="dev"):
    lamp_dataset_path = "datasets"
    os.makedirs(lamp_dataset_path, exist_ok=True)
    data_path = os.path.join(lamp_dataset_path, f"lamp_{dataset_num}_{mode}_data.pkl")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        with urllib.request.urlopen(f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{dataset_num}/{mode}/{mode}_questions.json") as url:
            data = json.load(url)
            data = sorted(data, key=lambda x: int(x["id"]))
        with open(data_path, "wb") as f:
            pickle.dump(data, f)
    gts_path = os.path.join(lamp_dataset_path, f"lamp_{dataset_num}_{mode}_gts.pkl")
    if os.path.exists(gts_path):
        with open(gts_path, "rb") as f:
            gts = pickle.load(f)
    else:
        with urllib.request.urlopen(f"https://ciir.cs.umass.edu/downloads/LaMP/LaMP_{dataset_num}/{mode}/{mode}_outputs.json") as url:
            gts = json.load(url)["golds"]
            gts = sorted(gts, key=lambda x: int(x["id"]))
        with open(gts_path, "wb") as f:
            pickle.dump(gts, f)
    return data, gts

def get_profvar_names(dataset_num):
    if dataset_num == 5:
        prof_gt_name = "title"
        prof_text_name = "abstract"
        prof_prompt_name = "abstract"
    elif dataset_num == 3:
        prof_gt_name = "score"
        prof_text_name = "text"
        prof_prompt_name = "review"
    return prof_text_name, prof_gt_name, prof_prompt_name

def create_retr_data(data, dataset_num=5):
    queries = []
    profile_text = []
    profile_gts = []
    prof_text_name, prof_gt_name, _  = get_profvar_names(dataset_num)
    for sample in data:
        text_idx = sample["input"].find(":") + 1
        queries.append(sample["input"][text_idx:].strip())
        profile_gts.append([p[prof_gt_name] for p in sample["profile"]])
        profile_text.append([p[prof_text_name] for p in sample["profile"]])
    return queries, profile_text, profile_gts
    """
    query_lens = pd.Series([len(query.split(" ")) for query in queries])
    query_len_cutoff = query_lens.quantile(0.995)
    outgts_idx = []
    for i, q in enumerate(queries):
        if len(q.split(" ")) > query_len_cutoff:
            outgts_idx.append(i)
    queries = [i for j, i in enumerate(queries) if j not in outgts_idx]
    out_gts = [i for j, i in enumerate(out_gts) if j not in outgts_idx]
    profile_text = [i for j, i in enumerate(profile_text) if j not in outgts_idx]
    profile_gts = [i for j, i in enumerate(profile_gts) if j not in outgts_idx]
    text_lens = [[len(t.split(" ")) for t in text] for text in profile_text]
    text_lens = pd.Series(list(chain.from_iterable(text_lens)))
    text_lens_cutoff = text_lens.quantile(0.995)
    for ic, text in enumerate(profile_text):
        out_idx = []
        for i, c in enumerate(text):
            if len(c.split(" ")) > text_lens_cutoff or f"No {prof_text_name} available" in c:
                out_idx.append(i)
        profile_text[ic] = [i for j, i in enumerate(profile_text[ic]) if j not in out_idx]
        profile_gts[ic] = [i for j, i in enumerate(profile_gts[ic]) if j not in out_idx]
    return queries, profile_text, profile_gts, out_gts, outgts_idx
    """

def retrieved_idx(prof_text, queries, dataset_num, dataset_split, model="bm25", device="cuda:0"):
    retr_path = f"retrievers/{dataset_num}/{dataset_split}"
    os.makedirs(retr_path, exist_ok=True)
    file_path = os.path.join(retr_path, f"{model}.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            retr_doc_idxs = pickle.load(f)
    else:
        retr_doc_idxs = []
        if model == "bm25":
            for i in range(len(prof_text)):
                bm25 = BM25Okapi(prof_text[i])
                doc_scores = bm25.get_scores(queries[i])
                retr_doc_idxs.append(doc_scores.argsort()[::-1])
        elif model in ["contriever", "dpr"]:
            if model == "contriever":
                retr_model = Contriever.from_pretrained("facebook/contriever-msmarco") 
                tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
            elif model == "dpr":
                retr_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
                tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")  
            retr_model.to(device).eval()
            with torch.no_grad():
                for i in range(len(prof_text)):
                    inp = prof_text[i]
                    inp.append(queries[i]) 
                    inputs = tokenizer(inp, padding=True, truncation=True, return_tensors="pt")
                    inputs.to(device)
                    embeddings = retr_model(**inputs)
                    if model == "dpr":
                        embeddings = embeddings.pooler_output
                    embeddings = embeddings.cpu()
                    sim_scores = np.dot(embeddings[-1:], embeddings[:-1].T)    
                    sorted_idxs = np.argsort(sim_scores).squeeze()[::-1]
                    retr_doc_idxs.append(sorted_idxs)
        else:
            raise Exception("Retriever not implemented!")     
        with open(file_path, "wb") as f:
            pickle.dump(retr_doc_idxs, f)
    return retr_doc_idxs
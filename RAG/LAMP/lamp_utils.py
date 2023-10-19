import argparse
from itertools import chain
import os
import pickle

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from contriever.src.contriever import Contriever

from RAG.loader import FileLoader

def get_lamp_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("-isq", "--quant_bots", default=0, type=int)
   parser.add_argument("-dn", "--dataset_num", default="5", type=str)
   parser.add_argument("-k", "--k", default="0", type=str)
   parser.add_argument("-r", "--retriever", default="bm25", type=str)

   return parser.parse_args()

def get_val_idx(processed_gts, dataset_num=5):
    _, out_gts = FileLoader.get_lamp_dataset(dataset_num, ["dev"])
    out_gts = out_gts["dev"]
    val_idx = []
    for og in out_gts:
        try:
            val_idx.append(processed_gts.index(og))
        except ValueError:
            continue
    return val_idx

def create_retr_data(data, out_gts):
    queries = []
    corpuses = []
    titles = []
    for sample in data:
        abstract_idx = sample["input"].find(":") + 1
        queries.append(sample["input"][abstract_idx:].strip())
        titles.append([p["title"] for p in sample["profile"]])
        corpuses.append([p["abstract"] for p in sample["profile"]])
    query_lens = pd.Series([len(query.split(" ")) for query in queries])
    query_len_cutoff = query_lens.quantile(0.995)
    out_idx = []
    for i, q in enumerate(queries):
        if len(q.split(" ")) > query_len_cutoff:
            out_idx.append(i)
    queries = [i for j, i in enumerate(queries) if j not in out_idx]
    out_gts = [i for j, i in enumerate(out_gts) if j not in out_idx]
    corpuses = [i for j, i in enumerate(corpuses) if j not in out_idx]
    titles = [i for j, i in enumerate(titles) if j not in out_idx]
    corp_lens = [[len(corp.split(" ")) for corp in corpus] for corpus in corpuses]
    corp_lens = pd.Series(list(chain.from_iterable(corp_lens)))
    corp_lens_cutoff = corp_lens.quantile(0.995)
    for ic, corpus in enumerate(corpuses):
        out_idx = []
        for i, c in enumerate(corpus):
            if len(c.split(" ")) > corp_lens_cutoff or "No abstract available" in c:
                out_idx.append(i)
        corpuses[ic] = [i for j, i in enumerate(corpuses[ic]) if j not in out_idx]
        titles[ic] = [i for j, i in enumerate(titles[ic]) if j not in out_idx]
    return queries, corpuses, titles, out_gts

def retrieved_idx(corpuses, queries, model="bm25", device="cuda:0"):
    retr_path = "retrievers"
    os.makedirs(retr_path, exist_ok=True)
    file_path = os.path.join(retr_path, f"{model}.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            retr_doc_idxs = pickle.load(f)
    else:
        retr_doc_idxs = []
        if model == "bm25":
            for i in range(len(corpuses)):
                bm25 = BM25Okapi(corpuses[i])
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
                for i in range(len(corpuses)):
                    inp = corpuses[i]
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
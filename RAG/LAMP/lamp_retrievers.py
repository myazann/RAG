from rank_bm25 import BM25Okapi
import pandas as pd
from itertools import chain

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

def bm25(corpuses, queries):

    ret_docs = []
    for i in range(len(corpuses)):

        bm25 = BM25Okapi(corpuses[i])
        doc_scores = bm25.get_scores(queries[i])
        ret_docs.append(doc_scores.argsort()[::-1])

    return ret_docs
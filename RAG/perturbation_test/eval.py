import json
import pickle

from langsmith import Client
import huggingface_hub
from ragas import evaluate
from ragas.metrics import AnswerRelevancy
from datasets import Dataset

from RAG.utils import get_args, get_device
from RAG.output_formatter import find_best_substring_match
from RAG.chatbots import choose_bot

args = get_args()
device = get_device()
test = args.perturb_test_type

chatbot = choose_bot(device)
huggingface_hub.login(new_session=False)
client = Client()
perturb_tests = [x for x in client.list_projects() if x.name[:2] == "PT"]

all_qas = {}
with open(f"{test}.json", "r") as f:
    test_file = json.load(f) 

    for k, v in test_file.items():
        for questions in v["questions"]:
            all_qas[questions] = v["answer"]

all_test_res = {}

for perturb_test in perturb_tests:

    res_dataset = {
        "question": [],
        "ground_truths": [],
        "answer": [],
        "contexts": [],
        "source_doc_match_ratio": []
    }
    pt_test_name = perturb_test.name
    print(pt_test_name)
    k = pt_test_name.split("_")[3]
    model_name = pt_test_name.split("_")[-2]

    runs = list(client.list_runs(
        project_name=pt_test_name,
        execution_order=1,  
        error=False,  
    ))

    runs.reverse()

    for i, run in enumerate(runs):
        
        question = run.inputs["question"]
        ground_truth = [all_qas[question]]
        answer = run.outputs["answer"]
        source_docs = [page["page_content"] for page in run.outputs["source_documents"]]
        
        res_dataset["question"].append(question)
        res_dataset["ground_truths"].append(ground_truth)
        res_dataset["answer"].append(answer)
        res_dataset["contexts"].append(source_docs)
        match_ratio, _ = find_best_substring_match("\n".join(source_docs), ground_truth[0])
        res_dataset["source_doc_match_ratio"].append(match_ratio)

    dataset = Dataset.from_dict(res_dataset)

    answer_relevancy = AnswerRelevancy(llm=chatbot.pipe, strictness=3)
    result = evaluate(dataset, metrics=[answer_relevancy])
    df = result.to_pandas()
    df["source_doc_match_ratio"] = res_dataset["source_doc_match_ratio"]
    all_test_res[f"{model_name}_{k}"] = df

    print(df.head())
    
with open("llm_eval_res.pkl", "wb") as f:
    pickle.dump(all_test_res, f)
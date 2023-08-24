import json
import time
import os

import xlsxwriter
import pandas as pd
from langchain import LLMChain, PromptTemplate
from evaluate import load
from langsmith import Client

from RAG.utils import get_args, get_device
from RAG.prompter import Prompter
from RAG.output_formatter import find_best_substring_match, eval_output_formatter
from RAG.chatbots import choose_bot
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator 

args = get_args()
device = get_device()
test = args.perturb_test_type

prompter = Prompter()
chatbot = choose_bot(device, model_name="CLAUDE-V2", gen_params={"max_new_tokens": 512, "temperature": 0})
eval_prompt = prompter.merge_with_template(chatbot, "eval_qa")
llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(eval_prompt))

client = Client()
perturb_tests = [x for x in client.list_projects() if x.name[:2] == "PT"]

all_qas = {}
with open(f"{test}.json", "r") as f:
    test_file = json.load(f) 

    for k, v in test_file.items():
        for questions in v["questions"]:
            all_qas[questions] = v["answer"]

out_file = "llm_eval_res.xlsx"
if os.path.exists(out_file):
    df = pd.read_excel(out_file)
else:
    df = pd.DataFrame()
for perturb_test in perturb_tests:

    pt_test_name = perturb_test.name
    print(pt_test_name)
    k = pt_test_name.split("_")[3]
    model_name = pt_test_name.split("_")[-2]

    if len(df[(df["model_name"] == model_name) & (df["k"] == k)]) != 0:
        continue

    else:
        res_dataset = {
            "question": [],
            "ground_truths": [],
            "answer": [],
            "contexts": [],
            "source_doc_match_ratio": [],
            "bert_score_precision": [],
            "bert_score_recall": [],
            "bert_score_f1": [],
            "UniEval_consistency": [],
            "LLM_Correctness": [],
            "LLM_Relevance": [],
            "LLM_Coherence": [],
            "LLM_Analysis": []
        }

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

            eval_res = llm_chain.predict(question=question, solution=ground_truth[0], answer=answer)
            out_dict = eval_output_formatter(eval_res)
            res_dataset["LLM_Correctness"].append(out_dict["Correctness"]),
            res_dataset["LLM_Relevance"].append(out_dict["Relevance"]),
            res_dataset["LLM_Coherence"].append(out_dict["Coherence"]),
            res_dataset["LLM_Analysis"].append(out_dict["Analysis"]),
            
            res_dataset["question"].append(question)
            res_dataset["ground_truths"].append(ground_truth)
            res_dataset["answer"].append(answer)
            res_dataset["contexts"].append(source_docs)
            match_ratio, _ = find_best_substring_match("\n".join(source_docs), ground_truth[0])
            res_dataset["source_doc_match_ratio"].append(match_ratio)

            bertscore = load("bertscore")
            bertscore_res = bertscore.compute(predictions=[answer], references=ground_truth, lang="en")
            res_dataset["bert_score_precision"].append(bertscore_res["precision"])
            res_dataset["bert_score_recall"].append(bertscore_res["recall"])
            res_dataset["bert_score_f1"].append(bertscore_res["f1"])
            
            task = "fact"
            data = convert_to_json(output_list=[answer], src_list=ground_truth)
            evaluator = get_evaluator(task)
            eval_scores = evaluator.evaluate(data, print_result=False)
            res_dataset["UniEval_consistency"].append(eval_scores[0]["consistency"])

        test_df = pd.DataFrame(res_dataset)
        test_df["k"] = k
        test_df["model_name"] = model_name

        df = pd.concat([df, test_df])

        with pd.ExcelWriter("llm_eval_res.xlsx", engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
import json
import re
import os
import time

from langsmith import Client
from langchain import LLMChain, PromptTemplate
import huggingface_hub

from RAG.chatbots import choose_bot
from RAG.utils import get_args, get_device, find_best_substring_match
from RAG.prompter import Prompter
from RAG.output_formatter import eval_output_formatter

args = get_args()
device = get_device()
test = args.perturb_test_type

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
prompter = Prompter()
chatbot = choose_bot(device, gen_params={"max_new_tokens": 512, "temperature": 0})

for perturb_test in perturb_tests:

    pt_test_name = perturb_test.name
    print(pt_test_name)
    all_test_res[pt_test_name] = []
    ls_name = f"Eval_{pt_test_name}_{time.time()}"
    os.environ["LANGCHAIN_PROJECT"] = ls_name

    runs = list(client.list_runs(
        project_name=pt_test_name,
        execution_order=1,  
        error=False,  
    ))

    runs.reverse()
    
    eval_prompt = prompter.merge_with_template(chatbot, "eval_qa")
    llm_chain = LLMChain(llm=chatbot.pipe, prompt=PromptTemplate.from_template(eval_prompt))

    for run in runs:
        
        question = run.inputs["question"]
        solution = all_qas[question]
        answer = run.outputs["answer"]
        source_docs = " \n".join([page["page_content"] for page in run.outputs["source_documents"]])

        eval_res = llm_chain.predict(question=question, solution=solution, answer=answer)

        out_dict = eval_output_formatter(eval_res)

        source_doc_match_ratio, _ = find_best_substring_match(source_docs, solution)
        print()
        print(f"""Scores: {out_dict} \nSource doc match ratio: {source_doc_match_ratio}""")

        all_test_res[pt_test_name].append({
            "Question": question,
            "Solution": solution,
            "Answer": answer,
            "Correctness Score": out_dict["Correctness"],
            "Relevance Score": out_dict["Relevance"],
            "Coherence Score": out_dict["Coherence"],
            "Explanation": out_dict["Explanation"],
            "Source Doc Match Ratio": source_doc_match_ratio 
        })

with open(f"evalres_{test}.json", "w") as f:
    json.dump(all_test_res, f)
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

def format_output(output):

    start = output.find("{")
    end = output.find("}")
    output = output[start: end+1].strip().replace("'", '"')
    eval_dict = json.loads(output, strict=False)

    return {k.lower(): v for k, v in eval_dict.items()}

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
chatbot = choose_bot(device)

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
    llm_chain = LLMChain(
            llm=chatbot.pipe,
            prompt=PromptTemplate.from_template(eval_prompt)
        )

    for run in runs:
        
        question = run.inputs["question"]
        real_answer = all_qas[question]
        gen_answer = run.outputs["answer"]
        source_docs = " \n".join([page["page_content"] for page in run.outputs["source_documents"]])

        eval_res = llm_chain.predict(question=question, real_answer=real_answer, gen_answer=gen_answer)

        score_match = re.search(r"Score: (\d+)", eval_res)
        if score_match:
            score = int(score_match.group(1))

        exp_match = re.search(r"Explanation:\s*(.+)", eval_res, re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1)

        source_doc_match_ratio, _ = find_best_substring_match(source_docs, real_answer)
        print()
        print(f"""Score: {score}\nExplanation: {explanation} \nSource doc match ratio: {source_doc_match_ratio}""")

        all_test_res[pt_test_name].append({
            "Q": question,
            "RA": real_answer,
            "GA": gen_answer,
            "Score": score,
            "Explanation": explanation,
            "Source Doc Match Ratio": source_doc_match_ratio 
        })

# pretty_name = "_".join(test_name.split("_")[2:-1])
with open(f"evalres_{test}.json", "w") as f:
    json.dump(all_test_res, f)

"""
chatbot = choose_bot(device, gen_params={"max_new_tokens": 512, "temperature": 0})
lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

prompter = Prompter()
eval_prompt = prompter.merge_with_template(chatbot, "eval_qa")
eval_res = lc_pipeline(eval_prompt.format(question=question, real_answer=real_answer, gen_answer=gen_answer))



eval_dict = json.loads(eval_res)

score_match = re.search(r"Score: (\d+)", eval_res)
if score_match:
    score = int(score_match.group(1))

exp_match = re.search(r"Explanation:\s*(.+)", eval_res, re.DOTALL)
if exp_match:
    explanation = exp_match.group(1)

"""
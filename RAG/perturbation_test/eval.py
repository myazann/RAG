import json
import re

from langchain import HuggingFacePipeline

from RAG.chatbots import choose_bot
from RAG.utils import init_env, find_best_substring_match
from RAG.prompter import Prompter

args, device, _ = init_env("Eval")

with open("test1_res.json", "r") as f:
    perturb_res = json.load(f) 

chatbot = choose_bot(device, gen_params={"max_new_tokens": 512, "temperature": 0})
lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

prompter = Prompter()
eval_prompt = prompter.merge_with_template(chatbot, "eval_qa")

models = list(perturb_res.keys())
num_questions = len(perturb_res[models[0]]["Questions"])
for model in models:
    for i in range(num_questions):
        real_answer = perturb_res[model]["Real Answer"][i]
        for q, gen_a, source_doc in zip(perturb_res[model]["Questions"][i], perturb_res[model]["Generated Answers"][i], perturb_res[model]["Source Docs"][i]):

            eval_res = lc_pipeline(eval_prompt.format(question=q, real_answer=real_answer, gen_answer=gen_a))
            
            score_match = re.search(r"Score: (\d+)", eval_res)
            if score_match:
                score = int(score_match.group(1))

            exp_match = re.search(r'Explanation:\s*(.+)', eval_res)
            if exp_match:
                explanation = exp_match.group(1)


            source_doc_match_ratio, _ = find_best_substring_match(source_doc, real_answer)
            print(f"Score: {score}, Explanation: {explanation}, Source doc match ratio: {source_doc_match_ratio}")
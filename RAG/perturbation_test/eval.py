import json

from langchain import HuggingFacePipeline

from RAG.chatbots import choose_bot
from RAG.utils import init_env
from RAG.prompter import Prompter

args, device, _ = init_env("Eval")

with open("test1_res.json", "r") as f:
    perturb_res = json.load(f) 

chatbot = choose_bot(device)
lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

prompter = Prompter()
eval_prompt = prompter.merge_with_template(chatbot, "eval_qa")

models = list(perturb_res.keys())
num_questions = len(perturb_res[models[0]]["Questions"])
for model in models:
    for i in range(num_questions):
        real_answer = perturb_res[model]["Real Answer"][i]
        for q, gen_a in zip(perturb_res[model]["Questions"][i], perturb_res[model]["Generated Answers"][i]):

            res = lc_pipeline(eval_prompt.format(question=q, real_answer=real_answer, gen_answer=gen_a))
            print(f"Question: {q}")
            print(f"Generated Answer: {gen_a}")
            print(f"Real Answer: {real_answer}")
            print(res)
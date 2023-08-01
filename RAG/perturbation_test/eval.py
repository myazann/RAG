import json

from langchain import HuggingFacePipeline

from RAG.chatbots import choose_bot
from RAG.utils import init_env

args, device, _ = init_env("Eval")

eval_prompt = """I want you to act as an evaluator. I will give you a question, the real answer, and the generated answer. 
You will determine how much the generated answer is close to the real answer. Give a score between 0 and 100. Reason step 
by step about why the score is appropriate, then print the score at the end. At the end, repeat that score alone on a new line.
Question: {question}
Real answer: {real_answer}
Generated answer: {gen_answer}"""

with open("perturb_res.json", "r") as f:
    perturb_res = json.load(f) 

chatbot = choose_bot(device)
lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

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
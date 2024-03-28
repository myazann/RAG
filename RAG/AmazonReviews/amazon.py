import argparse
import os
import json
import time
import pandas as pd

from RAG.chatbots import choose_bot
from RAG.prompter import Prompter
from RAG.AmazonReviews.data_preprocessing import download_datasets, get_dfs, create_user_data

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cat", default="All_Beauty", type=str)
parser.add_argument("-q", "--quant", default=None, type=str)
parser.add_argument("-t", "--task", default="review", type=str)
parser.add_argument("-mt","--max_tokens", default=512, type=int)
parser.add_argument("-n","--n_turns", default=7, type=int)
args = parser.parse_args()

category = args.cat
max_tokens = args.max_tokens
n_turns = args.n_turns
task = args.task
q_type = args.quant

download_datasets(category)
df, df_meta = get_dfs(category)
all_user_data = create_user_data(df, df_meta, category)
print(pd.Series([len(all_user_data[u]["History"]) for u in all_user_data.keys()]).value_counts())
prompter = Prompter()
print(f"Total number of customers: {len(all_user_data.keys())}")

all_analysis = {}
chatbots = ["MISTRAL-7B-v0.2-INSTRUCT", "MISTRAL-8x7B-v0.1-INSTRUCT", "LLAMA2-70B"]
if q_type is not None:
    chatbots = [f"{bot}-{q_type}" for bot in chatbots]
for bot in chatbots:
    print(bot)
    chatbot = choose_bot(model_name=bot, gen_params={"max_tokens": max_tokens})
    for i, user in enumerate(all_user_data.keys()):
        print(i)
        all_prods = []
        cust_hist = ""
        for prod in  all_user_data[user]["History"]:
            all_cats = "\n".join(prod["Categories"]).strip()
            all_descs = "\n".join(prod["Descriptions"]).strip()
            prod_desc = f"Product Title:\n{prod['Name']}\nProduct Categories:\n{all_cats}\nProduct Descriptions:\n{all_descs}\nCustomer Review:\n{prod['Review']}\nCustomer Score:\n{prod['Score']}\n"
            all_prods.append(prod_desc)
        all_prods.reverse()
        cust_hist = "\n".join(all_prods)
        while True:
            if task == "data_gen":
                prompt = prompter.amazon_np_pred_with_conv_claude_cot(cust_hist=cust_hist)
            elif task == "review":
                prompt = prompter.amazon_review_gen(cust_hist=cust_hist, prod_name=all_user_data[user]["Product"]["Name"], rating=all_user_data[user]["Product"]["Score"])
            if chatbot.count_tokens(prompt) > int(chatbot.context_length) - max_tokens:
                all_prods = all_prods[:-1]
                cust_hist = "\n".join(all_prods)
            else: 
                break
        start_time = time.time()
        response = chatbot.prompt_chatbot(prompt)
        print(f"Took {time.time()-start_time} secs.")
        print()
        print(response)
        print()
        if task == "review":
            print(f"Ground Truth:\n{all_user_data[user]['Product']['Review']}")
        all_analysis[user] = response
    os.makedirs("Results", exist_ok=True)
    target_path = os.path.join("Results", f"{chatbot.model_name}_{category}_Reviews.json")
    with open(target_path, "w") as f:
        json.dump(all_analysis, f)
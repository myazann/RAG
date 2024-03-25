import argparse

from RAG.chatbots import choose_bot
from RAG.prompter import Prompter
from RAG.AmazonReviews.data_preprocessing import download_datasets, get_dfs, create_user_data

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cat", default="All_Beauty", type=str)
parser.add_argument("-mt","--max_tokens", default=1024, type=int)
parser.add_argument("-n","--n_turns", default=7, type=int)
args = parser.parse_args()

category = args.cat
max_tokens = args.max_tokens
n_turns = args.n_turns

download_datasets(category)
df, df_meta = get_dfs(category)
all_user_data = create_user_data(df, df_meta, category)
chatbot = choose_bot(gen_params={"max_tokens": max_tokens})
prompter = Prompter()
all_analysis = {}
print(f"Total number of customers: {len(all_user_data.keys())}")

for user in all_user_data.keys():
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
        prompt = prompter.amazon_np_pred_with_conv_claude_cot(cust_hist=cust_hist)
        if chatbot.count_tokens(prompt) > int(chatbot.context_length) - max_tokens:
            all_prods = all_prods[:-1]
            cust_hist = "\n".join(all_prods)
        else: 
            break
    print(chatbot.count_tokens(cust_hist))
    user_analysis = chatbot.prompt_chatbot(prompt)
    print(user_analysis)
    print()
    all_analysis[user] = user_analysis
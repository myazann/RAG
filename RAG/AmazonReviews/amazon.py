import requests
import os
import gzip
import pandas as pd
import json
import datetime

from RAG.chatbots import choose_bot
from RAG.prompter import Prompter

def download_datasets(category):
    os.makedirs("datasets", exist_ok=True)
    review_link = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFiles/{category}.json.gz"
    review_save_loc = os.path.join("datasets", f"{category}.json.gz")
    if os.path.exists(review_save_loc):
        print("Reviews for this category already exists!")
    else:
        print(f"Downloading reviews for the {category} category!")
        response = requests.get(review_link)
        with open(review_save_loc, 'wb') as f:
            f.write(response.content)
    meta_link = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_{category}.json.gz"
    meta_save_loc = os.path.join("datasets", f"{category}_meta.json.gz")
    if os.path.exists(meta_save_loc):
        print("Metadata for this category already exists!")
    else:
        print(f"Downloading metadata for the {category} category!")
        response = requests.get(meta_link)
        with open(meta_save_loc, 'wb') as f:
            f.write(response.content)    
            
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def get_dfs(category, process=True):
    df = getDF(os.path.join("datasets", f"{category}.json.gz"))
    df_meta = getDF(os.path.join("datasets", f"{category}_meta.json.gz"))
    if process:
        df = df[~df["reviewerName"].str.contains("ustomer", na=False)]
        df = df.drop_duplicates(["reviewerID", "unixReviewTime"])
        df = df[df["reviewerID"].isin(df.value_counts("reviewerID")[(df.value_counts("reviewerID") > 5)].index)]
        up_lim = df["reviewerID"].value_counts().quantile(q=0.99)
        df = df[df["reviewerID"].isin(df.value_counts("reviewerID")[(df.value_counts("reviewerID") < up_lim)].index)]
        df_meta = df_meta.drop_duplicates("asin")
    return df, df_meta

def create_user_data(df, df_meta, category):
    data_path = os.path.join("datasets", f"{category}_user_data.json")
    all_users = df["reviewerID"].unique()
    if not os.path.exists(data_path):
        all_user_data = {}
        start_idx = 0
    else:
        with open(data_path, "r") as f:
            all_user_data = json.load(f)
            if len(all_user_data) == len(all_users):
                print("User data for this category is already created!")
                return all_user_data
            else:
                start_idx = len(all_user_data)
    print("Processing user data...")
    for num_user, user_id in enumerate(all_users[start_idx:]):
        sample_user = df[df["reviewerID"] == user_id].sort_values("unixReviewTime")
        sample_user = sample_user.merge(right=df_meta[["asin", "brand", "title", "category", "description"]], on="asin", how="inner")
        user_data = {}
        i = 1
        user_history = []
        for _, row in sample_user.iterrows():
            date_time = datetime.datetime.fromtimestamp(row["unixReviewTime"])
            formatted_date = date_time.strftime("%Y-%m-%d %H:%M:%S")
            prod_info = {
                "Name": row['title'],
                "Categories": row["category"],
                "Descriptions": row["description"],
                "Review": row["reviewText"],
                "Score": row["overall"],
                "Review Time": formatted_date
            }
            if i == len(sample_user):
                user_data["Product"] = prod_info
            else:
                user_history.append(prod_info)
            i += 1
        user_data["History"] = user_history
        all_user_data[user_id] = user_data
        if (num_user+1) % 500 == 0:
            print(f"Step: {num_user+start_idx}")
            with open(data_path, "w") as f:
                json.dump(all_user_data, f)
    print("Finished processing user data!")
    with open(data_path, "w") as f:
        json.dump(all_user_data, f)
    return all_user_data

def run_prompt(chatbot, prompt_type="conv_gen"):
    prompter = Prompter()
    all_analysis = {}
    print(f"Total number of customers: {len(all_user_data.keys())}")
    for user in all_user_data.keys():
        cust_hist = ""
        for prod in all_user_data[user]["History"]:
            all_cats = "\n".join(prod["Categories"]).strip()
            all_descs = "\n".join(prod["Descriptions"]).strip()
            prod_desc = f"Product Title:\n{prod['Name']}\nProduct Categories:\n{all_cats}\nProduct Descriptions:\n{all_descs}\nCustomer Review:\n{prod['Review']}\nCustomer Score:\n{prod['Score']}\n"
            cust_hist = f"{cust_hist}\n{prod_desc}"
        if prompt_type == "cust_analysis":
            prompt = prompter.amazon_cust_analysis_prompt(cust_hist=cust_hist.strip())
        elif prompt_type == "conv_gen":
            prompt = prompter.amazon_np_pred_with_conv_claude_cot(cust_hist=cust_hist.strip())
        if chatbot.count_tokens(prompt) < int(chatbot.context_length):
            user_analysis = chatbot.prompt_chatbot(prompt)
            print(user_analysis)
            print()
            all_analysis[user] = user_analysis
        else:
            print("User has a very long history!")
            print(chatbot.count_tokens(prompt))
    return all_analysis

category = "All_Beauty"
download_datasets(category)
df, df_meta = get_dfs(category)
all_user_data = create_user_data(df, df_meta, category)

chatbot = choose_bot(gen_params={"max_tokens": 1024})
run_prompt(chatbot)
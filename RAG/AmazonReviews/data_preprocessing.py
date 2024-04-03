import requests
import os
import gzip
import pandas as pd
import json
import datetime

def download_datasets(category):
    os.makedirs("datasets", exist_ok=True)
    review_link = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/{category}.jsonl.gz" 
    review_save_loc = os.path.join("datasets", f"{category}.jsonl.gz")
    if os.path.exists(review_save_loc):
        print("Reviews for this category already exists!")
    else:
        print(f"Downloading reviews for the {category} category!")
        response = requests.get(review_link)
        with open(review_save_loc, 'wb') as f:
            f.write(response.content)
    meta_link = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_{category}.jsonl.gz"
    meta_save_loc = os.path.join("datasets", f"{category}_meta.jsonl.gz")
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
  return pd.DataFrame.from_dict(df, orient="index")

def get_dfs(category, process=True):
    df = getDF(os.path.join("datasets", f"{category}.jsonl.gz"))
    df_meta = getDF(os.path.join("datasets", f"{category}_meta.jsonl.gz"))
    if process:
        # df = df[~df["reviewerName"].str.contains("ustomer", na=False)]
        df = df.drop_duplicates(["user_id", "timestamp"])
        df = df[df["user_id"].isin(df.value_counts("user_id")[(df.value_counts("user_id") > 2)].index)]
        # up_lim = df["reviewerID"].value_counts().quantile(q=0.99)
        # df = df[df["reviewerID"].isin(df.value_counts("reviewerID")[(df.value_counts("reviewerID") < up_lim)].index)]
        df_meta = df_meta.drop_duplicates("parent_asin")
    return df, df_meta

def create_user_data(df, df_meta, category):
    data_path = os.path.join("datasets", f"{category}_user_data.json")
    all_users = df["user_id"].unique()
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
        sample_user = df[df["user_id"] == user_id].sort_values("timestamp").rename(columns={"title": "reviewTitle", "text": "reviewText"})
        sample_user = sample_user.merge(right=df_meta[["parent_asin", "store", "title", "categories", "description", "details"]], on="parent_asin", how="inner")
        sample_user = sample_user.rename(columns={"title": "productTitle"})
        if len(sample_user) == 1:
            continue
        user_data = {}
        i = 1
        user_history = []
        for _, row in sample_user.iterrows():
            date_time = datetime.datetime.fromtimestamp(timestamp=row["timestamp"]/1000)
            formatted_date = date_time.strftime("%Y-%m-%d %H:%M:%S")
            prod_info = {
                "Name": row["productTitle"],
                # "Categories": row["categories"] ,
                "Descriptions": row["description"],
                "Details": row["details"],
                "Review": row["reviewText"],
                "Score": row["rating"],
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
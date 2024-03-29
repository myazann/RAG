import time
import warnings
warnings.filterwarnings("ignore")
import os

from RAG.chatbots import choose_bot
from RAG.utils import get_args
from RAG.vectordb import VectorDB
from RAG.loader import FileLoader
from RAG.prompter import Prompter
from RAG.output_formatter import query_reform_formatter

args = get_args()
web_search = args.web_search
file_loader = FileLoader()
chatbot = choose_bot()
mixtral_bot = choose_bot(model_name="MISTRAL-8x7B-v0.1-INSTRUCT-PPLX")
prompter = Prompter()
db = VectorDB(file_loader)

print("\nHello! How may I assist you? \nPress 0 if you want to quit!\nIf you want to provide a document or a webpage to the chatbot, please only input the path to the file or the url without any other text!\n")
chat_history = []
while True:
  print("User: ")
  query = input().strip()
  hist_to_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chatbot.trunc_chat_history(chat_history)])
  QUERY_GEN_PROMPT = prompter.query_gen_prompt_claude(chat_history=hist_to_str, user_input=query)
  if query == "0":
    print("Bye!")
    break
  else:
    reform_query = ""
    retr_docs = []
    info = ""
    start_time = time.time()
    if file_loader.get_file_type(query) in ["pdf", "git", "url"]:
      if not os.path.exists(query):
        print("Could not find file!")
        continue
      db.add_file_to_db(query)
      reform_query = query
      print(f"Time passed processing file: {round(time.time()-start_time, 2)} secs")
    else:
      reform_query = query_reform_formatter(mixtral_bot.prompt_chatbot(QUERY_GEN_PROMPT).strip())
      print(reform_query)
      if web_search:
        if "NO QUERY" not in reform_query:
          all_web_queries = mixtral_bot.prompt_chatbot(prompter.multi_query_prompt(question=reform_query)).strip()
          search_urls = file_loader.web_search(all_web_queries.split("\n"))
          db.add_file_to_db(search_urls)
          print(f"Time passed in web search: {round(time.time()-start_time, 2)} secs")
    all_db_docs = db.query_db()["documents"]
    if all_db_docs:
      # k = chatbot.find_best_k(all_db_docs)
      k = 10
      if reform_query == "":
        reform_query = query_reform_formatter(mixtral_bot.prompt_chatbot(QUERY_GEN_PROMPT).strip())
      if "NO QUERY" not in reform_query:
        retr_docs, distances, metadatas = db.query_db(query=reform_query, k=k, distance_threshold=0.75)
        print(distances)
        print(metadatas)
        print(f"Time passed in retrieval: {round(time.time()-start_time, 2)} secs")
    info = ""
    while True:
      if retr_docs:
        info = "\n".join([doc for doc in retr_docs])
      CONV_CHAIN_PROMPT = prompter.conv_agent_prompt(user_input=query, info=info)
      if chatbot.count_tokens(CONV_CHAIN_PROMPT) > int(chatbot.context_length):
        print("Context exceeds context window, removing one document!")
        retr_docs = retr_docs[:-1]
      else:
        break
    print(f"Time passed until generation: {round(time.time()-start_time, 2)} secs!")
    answer = chatbot.prompt_chatbot(CONV_CHAIN_PROMPT, chat_history)
    print("\nChatbot:")
    chatbot.stream_output(answer)
    end_time = time.time()
    print(f"\nTook {round(end_time - start_time, 2)} secs!\n")
    current_conv = f"""User: {query}\nAssistant: {answer}"""
    chat_history.extend([
                        {
                            "role": "user",
                            "content": query
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }])
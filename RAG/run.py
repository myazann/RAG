import time
import warnings
warnings.filterwarnings("ignore")
import os

from RAG.chatbot import choose_bot
from RAG.utils import get_args
from RAG.vectordb import VectorDB
from RAG.file_loader import FileLoader
from RAG.prompter import Prompter
from RAG.orchestrator import Orchestrator

args = get_args()
web_search = args.web_search
file_loader = FileLoader()
# model_params={"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
chatbot = choose_bot()
query_bot = choose_bot(model_name="LLAMA-3-70B-PPLX")
prompter = Prompter()
orchestrator = Orchestrator(query_bot)
db = VectorDB(file_loader)

print("\nHello! How may I assist you? \nPress 0 if you want to quit!\nPress -1 if you want to switch the chatbot!\nPress 1 if you want to clear chat history!\nIf you want to provide a document or a webpage to the chatbot, please only input the path to the file or the url without any other text!\n")
chat_history = []
while True:
    print("User: ")
    query = input().strip()
    hist_to_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chatbot.trunc_chat_history(chat_history)])
    QUERY_GEN_PROMPT = prompter.query_gen_prompt_claude(query=query, chat_history=hist_to_str)
    if query == "0":
        print("Bye!")
        break
    elif query == "-1":
        chatbot = choose_bot()
        continue
    elif query == "1":
        chat_history = []
        print("History cleared!")
        db = VectorDB(file_loader)
        continue
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
            if web_search:
                reform_query = orchestrator.decide_on_query(QUERY_GEN_PROMPT)
            if reform_query:
                print(reform_query)
                search_urls = file_loader.web_search(reform_query)
                print(search_urls)
                db.add_file_to_db(search_urls)
                print(f"Time passed in web search: {round(time.time()-start_time, 2)} secs")
            all_db_docs = db.query_db()["documents"]
            if all_db_docs:
                k = 10
                if reform_query == "":
                    reform_query = orchestrator.decide_on_query(QUERY_GEN_PROMPT)
                if reform_query:
                    retr_docs, distances, metadatas = db.query_db(query=reform_query, k=k, distance_threshold=0.75)
                    print(distances)
                    print(f"Time passed in retrieval: {round(time.time()-start_time, 2)} secs")
            info = ""
            CONV_CHAIN_PROMPT = prompter.conv_agent_prompt(query=query, context=info)
            if retr_docs:
                info = chatbot.prep_context(CONV_CHAIN_PROMPT, retr_docs, chat_history)
            CONV_CHAIN_PROMPT = prompter.conv_agent_prompt(query=query, context=info)
            print(f"Time passed until generation: {round(time.time()-start_time, 2)} secs!")
            answer = chatbot.prompt_chatbot(CONV_CHAIN_PROMPT, chat_history).strip()
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
import time
import os
import warnings
warnings.filterwarnings("ignore")

from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline, CohereRerank
import huggingface_hub

from RAG.chatbots import choose_bot
from RAG.utils import get_args
from RAG.vectordb import VectorDB
from RAG.loader import FileLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter
from RAG.output_formatter import query_reform_formatter, remove_exc_output

huggingface_hub.login(new_session=False)
args = get_args()
web_search = args.web_search
file_loader = FileLoader()
chatbot = choose_bot()
prompter = Prompter()
if chatbot.q_bit is None:
  test_name = f"QA_{chatbot.model_name}_{time.time()}"
else:
  test_name = f"QA_{chatbot.model_name}_{chatbot.q_bit}-bit_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name
db = VectorDB(file_loader)
emdeb_filter = EmbeddingsFilter(embeddings=db.get_embed_func("hf_bge"), similarity_threshold=0.8)
compressor = CohereRerank(cohere_api_key="RchaCL6jeh0FAazvWfB2G1qmAWNHeQiF3Qmg9ANO")
pipeline_compressor = DocumentCompressorPipeline(transformers=[emdeb_filter, compressor])
print("\nHello! How may I assist you? \nPress 0 if you want to quit!\nIf you want to provide a document or a webpage to the chatbot, please only input the path to the file or the url without any other text!\n")
summary = ""
while True:
  print("User: ")
  query = input().strip()
  QUERY_GEN_PROMPT = prompter.query_gen_prompt(summary=summary, user_input=query)
  if query == "0":
    print("Bye!")
    break
  else:
    if summary != "":
      MEMORY_PROMPT = prompter.memory_summary(summary=summary, new_lines=current_conv)
      summary = chatbot.prompt_chatbot(MEMORY_PROMPT).strip()
    reform_query = ""
    retr_docs = []
    info = ""
    added_files = []
    start_time = time.time()
    if file_loader.get_file_type(query) in ["pdf", "git", "url"]:
      if query not in added_files:
        db.add_file_to_db(query, web_search=False)
        added_files.append(query)
        reform_query = query
      else:
        print("File already in the system!")
      print(f"Time passed processing file: {round(time.time()-start_time, 2)} secs")
    elif web_search:
      reform_query = query_reform_formatter(chatbot.model_name, chatbot.pipe(QUERY_GEN_PROMPT).strip())
      if "NO QUERY" not in reform_query:
        db.add_file_to_db(reform_query, web_search)
        print(f"Time passed in web search: {round(time.time()-start_time, 2)} secs")
    all_db_docs = db.query_db()["documents"]
    if all_db_docs:
      k = chatbot.find_best_k(all_db_docs)
      retriever = Retriever(db.vector_db, k=k, comp_pipe=pipeline_compressor)
      if reform_query == "":
        reform_query = query_reform_formatter(chatbot.model_name, chatbot.pipe(QUERY_GEN_PROMPT).strip())
      if "NO QUERY" not in reform_query:
        retr_docs = retriever.get_docs(reform_query)
        print(f"Time passed in retrieval: {round(time.time()-start_time, 2)} secs")
    while True:
      if retr_docs:
        for i, doc in enumerate(retr_docs):
          info = "\n".join([doc.page_content for doc in retr_docs])
      CONV_CHAIN_PROMPT = prompter.conv_agent_prompt(user_input=query, chat_history=summary, info=info)
      if chatbot.count_tokens(CONV_CHAIN_PROMPT) > int(chatbot.context_length):
        if not retr_docs:
          raise Exception("The prompt is too long for the chosen chatbot!")
        print("Context exceeds context window, removing one document!")
        retr_docs = retr_docs[:-1]
      else:
        break
    print(f"Time passed until generation: {round(time.time()-start_time, 2)} secs!")
    answer = chatbot.prompt_chatbot(CONV_CHAIN_PROMPT)
    answer = remove_exc_output(chatbot.model_name, answer.content)
    print("\nChatbot:")
    chatbot.stream_output(answer)
    end_time = time.time()
    print(f"\nTook {round(end_time - start_time, 2)} secs!\n")
    current_conv = f"""User: {query}\nAssistant: {answer}"""
    print(current_conv)
    if retr_docs:
      print("Source of retrieved documents: ")
      for doc in retr_docs:
        print(doc.metadata)
        print(f"Similarity score: {doc.state['query_similarity_score']}")
import time
import os

from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
import huggingface_hub

from RAG.chatbots import choose_bot
from RAG.utils import get_args
from RAG.vectordb import VectorDB
from RAG.loader import FileLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter

huggingface_hub.login(new_session=False)
args = get_args()
web_search = args.web_search
file_loader = FileLoader()
chatbot = choose_bot()
prompter = Prompter()
if chatbot.q_bit is None:
  test_name = f"QA_{chatbot.name}_{time.time()}"
else:
  test_name = f"QA_{chatbot.name}_{chatbot.q_bit}-bit_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name
conv_agent_prompt = chatbot.prompt_chatbot(prompter.conv_agent_prompt())
memory_prompt = chatbot.prompt_chatbot(prompter.memory_summary())
db = VectorDB(file_loader)
emdeb_filter = EmbeddingsFilter(embeddings=db.get_embed_func("hf_bge"), similarity_threshold=0.75)
pipeline_compressor = DocumentCompressorPipeline(transformers=[emdeb_filter])
print("\nHello! How may I assist you? \nPress 0 if you want to quit!\nIf you want to provide a document or a webpage to the chatbot, please only input the path to the file or the url without any other text!\n")
summary = ""
while True:
  print("User: ")
  query = input().strip()
  if query == "0":
    print("Bye!")
    break
  else:
    start_time = time.time()
    if web_search:
      db.add_file_to_db(query, web_search)
    all_db_docs = db.query_db()["documents"]
    if all_db_docs:
      k = chatbot.find_best_k(all_db_docs, conv_agent_prompt)
      retriever = Retriever(db.vector_db, k=k, comp_pipe=pipeline_compressor)
      retr_docs = retriever.get_docs(query)
    else:
      retr_docs = []
    info = ""
    while True:
      if retr_docs:
        for i, doc in enumerate(retr_docs):
          info = "\n".join([doc.page_content for doc in retr_docs])
      CONV_CHAIN_PROMPT = conv_agent_prompt.format(user_input=query, chat_history=summary, info=info)
      if chatbot.count_tokens(CONV_CHAIN_PROMPT) > int(chatbot.context_length):
        print("Context exceeds context window, removing one document!")
        retr_docs = retr_docs[:-1]
      else:
        break
    answer = chatbot.pipe(CONV_CHAIN_PROMPT).strip()
    current_conv = f"""User: {query}\nAssistant: {answer}"""
    MEMORY_PROMPT = memory_prompt.format(summary=summary, new_lines=current_conv)
    summary = chatbot.pipe(MEMORY_PROMPT).strip()
    print("\nChatbot:")
    print(f"{answer}\n")
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")
    if retr_docs:
      print("Source of retrieved documents: ")
      for doc in retr_docs:
        print(doc.metadata)
        print(f"Similarity score: {doc.state['query_similarity_score']}")
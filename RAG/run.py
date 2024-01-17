import time
import os

from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import huggingface_hub

from RAG.chatbots import choose_bot
from RAG.utils import get_args, get_device
from RAG.loader import FileLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter

huggingface_hub.login(new_session=False)
args = get_args()
file_name = args.document
web_search = args.web_search
file_loader = FileLoader()
device = get_device()
chatbot = choose_bot()
prompter = Prompter()
if chatbot.q_bit is None:
  test_name = f"QA_{chatbot.name}_{time.time()}"
else:
  test_name = f"QA_{chatbot.name}_{chatbot.q_bit}-bit_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name
docs = []
conv_agent_prompt = chatbot.prompt_chatbot(prompter.conv_agent_prompt())
memory_prompt = chatbot.prompt_chatbot(prompter.memory_summary())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
model_name = "BAAI/bge-base-en"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
fs = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings, fs, namespace=embeddings.model_name
)
db = Chroma(embedding_function=embeddings)
emdeb_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.7)
pipeline_compressor = DocumentCompressorPipeline(transformers=[emdeb_filter])
if file_name is not None:
  print("Processing the file!")
  file, _ = file_loader.load(file_name)
  print("Done!")
  text_chunks = file_loader.get_processed_texts(text_splitter, file)
  last_id = len(docs)
  for i, chunk in enumerate(text_chunks):
    docs.append({"id": str(i), "content": chunk, "metadata": {"source": file_name}})
  db.add_texts(ids=[doc["id"] for doc in docs], texts=[doc["content"] for doc in docs], metadatas=[doc["metadata"] for doc in docs])
print("\nHello! How may I assist you? (Press 0 if you want to quit!)\n")
summary = ""
while True:
  print("User: ")
  query = input().strip()
  if query != "0":
    start_time = time.time()
    if web_search:
      web_texts, _ = file_loader.web_search(query)
      text_chunks = file_loader.get_processed_texts(text_splitter, web_texts)
      last_id = len(docs)
      for i, chunk in enumerate(text_chunks):
        docs.append({"id": str(i+last_id), "content": chunk, "metadata": {"source": "Web Search"}})
      db.add_texts(ids=[doc["id"] for doc in docs], texts=[doc["content"] for doc in docs], metadatas=[doc["metadata"] for doc in docs])
    k = chatbot.find_best_k([doc["content"] for doc in docs], conv_agent_prompt)
    retriever = Retriever(db, k=k, comp_pipe=pipeline_compressor)
    if not docs:
      CONV_CHAIN_PROMPT = conv_agent_prompt.format(user_input=query, chat_history=summary, info="")
    else:
      retr_docs = retriever.get_docs(query)
      while True:
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
    print("Source of retrieved documents: ")
    for doc in retr_docs:
      print(doc.metadata)
      print(f"Similarity score: {doc.state['query_similarity_score']}")
  else:
    print("Bye!")
    break
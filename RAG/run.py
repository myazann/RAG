import time
import os

from langchain.chains import ConversationChain
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import huggingface_hub

from RAG.chatbots import choose_bot
from RAG.utils import get_args, get_device
from RAG.loader import FileLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter
from RAG.output_formatter import csv_output_formatter

huggingface_hub.login(new_session=False)
args = get_args()
file_name = args.document
device = get_device()
chatbot = choose_bot()
file_loader = FileLoader()
file = file_loader.load(file_name)
file_type = file_loader.get_file_type(file_name)
prompter = Prompter()
if chatbot.q_bit is None:
  test_name = f"QA_{chatbot.name}_{time.time()}"
else:
  test_name = f"QA_{chatbot.name}_{chatbot.q_bit}-bit_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name
if file_type == "db":
  db_chain = SQLDatabaseChain.from_llm(chatbot.pipe, file, verbose=True)
elif file_type == "csv":
  df = file
  csv_prompt = chatbot.prompt_chatbot(prompter.csv_prompt())
  CSV_PROMPT = PromptTemplate(input_variables=["chat_history", "user_input"], template=csv_prompt)
  csv_chain = ConversationChain(llm=chatbot.pipe, input_key="user_input", 
                                memory=ConversationBufferWindowMemory(k=3, memory_key="chat_history"), prompt=CSV_PROMPT)
else:
  doc = file_loader.remove_empty_space(file)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
  texts = text_splitter.split_documents(doc)
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
  # embeddings = HuggingFaceEmbeddings()
  qa_prompt = chatbot.prompt_chatbot(prompter.qa_prompt())
  memory_prompt = chatbot.prompt_chatbot(prompter.memory_summary())
  db = Chroma.from_documents(texts, cached_embedder)
  emdeb_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.6)
  # context_comp_filter = LLMChainExtractor.from_llm(chatbot.pipe)
  pipeline_compressor = DocumentCompressorPipeline(transformers=[emdeb_filter])
  k = chatbot.find_best_k([page.page_content for page in texts], qa_prompt)
  retriever = Retriever(db, k=k, comp_pipe=pipeline_compressor)
# pretty_doc_name = " ".join(file_name.split(".")[:-1]).replace("_"," ")
print(f"""\nHello, I am here to inform you about the {file_name}. What do you want to learn? (Press 0 if you want to quit!) \n""")
summary = ""
while True:
  print("User: ")
  query = input().strip()
  if query != "0":
    start_time = time.time()
    if file_type == "db":
      answer = db_chain.run(query)
    elif file_type == "csv":
      col_info = df.dtypes.to_string()
      first_rows = df.head(5).to_string()
      query = f"Column names and datatypes:\n{col_info}\nFirst 5 rows of the dataframe:\n{first_rows}\nUser Input: {query}"
      answer = csv_chain.predict(user_input=query).strip()
      code = csv_output_formatter(answer)
      try:
        exec(code)
        answer = ""
      except Exception as e:
        print(f"Got an error for the chatbot generated code:\n {code}")
        answer = e
    else:
      retr_docs = retriever.get_docs(query)
      print(retr_docs[0])
      context = "\n".join([doc.page_content for doc in retr_docs])
      QA_CHAIN_PROMPT = qa_prompt.format(question=query, chat_history=summary, context=context)
      if chatbot.count_tokens(QA_CHAIN_PROMPT) > int(chatbot.context_length):
        print("Context exceeds context window, removing one document!")
        context = "\n".join([doc.page_content for doc in retr_docs[:-1]])
        QA_CHAIN_PROMPT = qa_prompt.format(question=query, chat_history=summary, context=context)
      answer = chatbot.pipe(QA_CHAIN_PROMPT).strip()
      current_conv = f"""Human: {query}\nAI: {answer}"""
      MEMORY_PROMPT = memory_prompt.format(summary=summary, new_lines=current_conv)
      summary = chatbot.pipe(MEMORY_PROMPT).strip()
    print("\nChatbot:")
    print(f"{answer}\n")
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")
  else:
    print("Bye!")
    break
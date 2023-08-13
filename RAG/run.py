import time
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryMemory
import huggingface_hub

from RAG.chatbots import choose_bot
from RAG.utils import get_args, get_device, get_NoOpChain
from RAG.loader import FileLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter

huggingface_hub.login(new_session=False)
args = get_args()
file_name = args.document
device = get_device()

chatbot = choose_bot(device)

file_loader = FileLoader()
file = file_loader.load(file_name)
file_type = file_loader.get_file_type(file_name)

test_name = f"QA_{chatbot.name}_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name

if file_type == "db":
  db_chain = SQLDatabaseChain.from_llm(chatbot.pipe, file, verbose=True)
else:
  doc = file_loader.trim_doc(file)

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
  texts = text_splitter.split_documents(doc)

  embeddings = HuggingFaceEmbeddings()
  db = Chroma.from_documents(texts, embeddings)

  retriever = Retriever(db)
  k = retriever.find_ideal_k(chatbot, [page.page_content for page in texts])
  retriever.init_base_retriever(k=k)
  retriever.add_embed_filter(embeddings, similarity_threshold=0.2)
  retriever.init_comp_retriever()
  # retriever.add_doc_compressor(chatbot.pipe)

  prompter = Prompter()
  qa_prompt = prompter.merge_with_template(chatbot, "qa")
  memory_prompt = prompter.merge_with_template(chatbot, "memory_summary")

  QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"], template=qa_prompt)
  MEMORY_PROMPT = PromptTemplate(input_variables=["summary", "new_lines"], template=memory_prompt)

  memory = ConversationSummaryMemory(llm=chatbot.pipe, memory_key="chat_history", return_messages=False, prompt=MEMORY_PROMPT,
                                     input_key="question", output_key="answer")

  doc_chain = load_qa_chain(
      chatbot.pipe,
      chain_type="stuff",
      **{"prompt": QA_CHAIN_PROMPT},
  )

  qa = ConversationalRetrievalChain(retriever=retriever.comp_retriever, combine_docs_chain=doc_chain, 
                                    question_generator=get_NoOpChain(chatbot.pipe), memory=memory, get_chat_history=lambda h: h,
                                    return_source_documents=True)

pretty_doc_name = " ".join(file_name.split(".")[:-1]).replace("_"," ")
print(f"""\nHello, I am here to inform you about the {pretty_doc_name}. What do you want to learn? (Press 0 if you want to quit!) \n""")

while True:
  print("User: ")
  query = input()
  if query != "0":
    start_time = time.time()
    if file_type == "db":
      answer = db_chain.run(query)
    else:
      result = qa({"question": query.strip()})
      answer = result["answer"].strip()
    print("\nChatbot:")
    print(f"{answer}\n")
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")
  else:
    print("Bye!")
    break
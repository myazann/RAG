import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline

from chatbots import choose_bot
from utils import init_env, get_args
from doc_loader import DocumentLoader

args, device = init_env("Document_QA")

chatbot = choose_bot(device)
lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

loader = DocumentLoader(args.document)
doc = loader.load_doc()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma")
db.persist()

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
qa = ConversationalRetrievalChain.from_llm(lc_pipeline, retriever, chain_type="stuff", return_source_documents=False)

chat_history = []

print("""\nHello, I am here to inform you about the LESSEN project. What do want to learn about LESSEN? (Press 0 if you want to quit!) \n""")

while True:
  query = input()
  if query != "0":
    start_time = time.time()
    result = qa({"question": query, "chat_history": chat_history})

    answer = result["answer"].strip()
    print(f"\n{answer}\n")
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")
    chat_history.append((query, answer))
  else:
    print("Bye!")
    break

## Who are the leaders of each work package in the LESSEN project?

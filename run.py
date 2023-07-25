import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline

from chatbots import choose_bot
from utils import get_device

## ls__7eb356bde9434566bcbcac0b9ee5844b

device = get_device()

pipe = choose_bot(device)
llm = HuggingFacePipeline(pipeline=pipe)

loader = PyPDFLoader("LESSEN_Project_Proposal.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
# embeddings = SentenceTransformerEmbeddings()


#db = DeepLake.from_documents(texts, embeddings)
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma")
db.persist()
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})


qa = ConversationalRetrievalChain.from_llm(llm, retriever, chain_type="stuff", return_source_documents=True)
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory, return_source_documents=True)

chat_history = []

print("""\nHello, I am here to inform you about the LESSEN project. What do want to learn about LESSEN? (Press 0 if you want to quit!) \n""")

while True:
  query = input()
  if query != "0":
    start_time = time.time()
    #result = qa({"question": query})
    result = qa({"question": query, "chat_history": chat_history})
    
    source_docs = result['source_documents']
    print(f"Source documents ({len(source_docs)}): \n")
    for docs in source_docs:
      print(docs)
    
    answer = result["answer"].strip()
    print(f"\n{answer}\n")
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")
    chat_history.append((query, answer))
  else:
    print("Bye!")
    break

## Who are the leaders of each work package in the LESSEN project?

import time
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from RAG.chatbots import choose_bot
from RAG.utils import init_env, get_args, get_device
from RAG.doc_loader import DocumentLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter

init_env()
args = get_args()
doc_name = args.document

loader = DocumentLoader(doc_name)
doc = loader.load_doc()
doc = loader.trim_doc(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
texts = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings)

chatbot = choose_bot(get_device())
lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

test_name = f"QA_{chatbot.repo.name}_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name

retriever = Retriever(db, k=3, search_type="mmr")
retriever.add_embed_filter(embeddings)
retriever.add_doc_compressor(lc_pipeline)

prompter = Prompter()
qa_prompt = prompter.merge_with_template(chatbot, "qa")
condense_prompt = prompter.merge_with_template(chatbot, "condense")
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=qa_prompt)
CONDENSE_PROMPT = PromptTemplate.from_template(condense_prompt)

qa = ConversationalRetrievalChain.from_llm(lc_pipeline, retriever.base_retriever, chain_type="stuff", 
                                           combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}, 
                                           condense_question_prompt=CONDENSE_PROMPT
                                           )

chat_history = []

pretty_doc_name = " ".join(doc_name.split(".")[:-1]).replace("_"," ")
print(f"""\nHello, I am here to inform you about the {pretty_doc_name} document. What do want to learn? (Press 0 if you want to quit!) \n""")

while True:
  query = input()
  if query != "0":
    start_time = time.time()
    result = qa({"question": query.strip(), "chat_history": chat_history})
    answer = result["answer"].strip()
    print(f"\n{answer}\n")
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")
    chat_history.append((query, answer))
  else:
    print("Bye!")
    break

## Who are the leaders of work packages?
## Give me a summary of each lessen work package
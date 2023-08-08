import time
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
import huggingface_hub

from RAG.chatbots import choose_bot
from RAG.utils import get_args, get_device
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

test_name = f"QA_{chatbot.repo.name}_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name

if file_type == "db":
  db_chain = SQLDatabaseChain.from_llm(chatbot.pipe, file, verbose=True)
else:
  doc = file_loader.trim_doc(file)

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
  texts = text_splitter.split_documents(doc)

  embeddings = HuggingFaceEmbeddings()

  db = Chroma.from_documents(texts, embeddings)

  retriever = Retriever(db, k=3, search_type="mmr")
  # retriever.add_embed_filter(embeddings)
  # retriever.add_doc_compressor(chatbot.pipe)

  prompter = Prompter()
  qa_prompt = prompter.merge_with_template(chatbot, "qa")
  condense_prompt = prompter.merge_with_template(chatbot, "condense")
  QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=qa_prompt)
  CONDENSE_PROMPT = PromptTemplate.from_template(condense_prompt)

  qa = ConversationalRetrievalChain.from_llm(chatbot.pipe, retriever.base_retriever, chain_type="stuff", 
                                            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}, 
                                            condense_question_prompt=CONDENSE_PROMPT
                                            )
  chat_history = []

pretty_doc_name = " ".join(file_name.split(".")[:-1]).replace("_"," ")
print(f"""\nHello, I am here to inform you about the {pretty_doc_name}. What do want to learn? (Press 0 if you want to quit!) \n""")

while True:
  query = input()
  if query != "0":
    start_time = time.time()
    if file_type == "db":
      answer = db_chain.run(query)
    else:
      result = qa({"question": query.strip(), "chat_history": chat_history})
      answer = result["answer"].strip()
      chat_history.append((query, answer))

    print(f"\n{answer}\n")
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")
  else:
    print("Bye!")
    break

## Who are the leaders of work packages?
## Give me a summary of each lessen work package


#I want you to act as an evaluator. I will give you a question, the solution, and the prediction, and you will give a score between 0 and 100 to the prediction. You will evaluate whether the prediction is similar to the solution and relevant to the question. The prediction does not have to exactly be the same as the solution, but the general meaning and context should be similar, and it should include include the information given in the solution. You should take into account whether the prediction goes off topic, repeats the same sentences over and over again, or contains unrelated, not mentioned or false information. False information and mention of unrelated information should be your priority, those answers should have a low score. If the prediction does not answer the question but is still trying to be helpful or polite, give it a score of 25. Your output will be as follows:
#        Score: <score>
#        Explanation: <your explanation about why you gave that score> 
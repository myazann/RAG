import json
import time
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate

import torch

from RAG.chatbots import choose_bot
from RAG.utils import init_env, get_device, get_args
from RAG.doc_loader import DocumentLoader
from RAG.retriever import Retriever
from RAG.enums import REPO_ID
from RAG.prompter import Prompter

init_env()
args = get_args()
doc_name = args.document
test = args.perturb_test_type

loader = DocumentLoader(doc_name)
doc = loader.load_doc()
doc = loader.trim_doc(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
texts = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings)

chatbots = [REPO_ID.VICUNA_13B_GPTQ, REPO_ID.LLAMA2_7B_GPTQ, REPO_ID.LLAMA2_13B_GPTQ, 
            REPO_ID.STABLE_BELUGA_7B_GPTQ, REPO_ID.STABLE_BELUGA_13B_GPTQ] 

with open(f"{test}.json", "r") as f:
    test_queries = json.load(f)

for bot in chatbots:

    print(bot.name)

    device = get_device()
    chatbot = choose_bot(device, bot, gen_params={"max_new_tokens": 512, "temperature": 0})
    lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

    test_name = f"PT_{test}_{bot.name}_{time.time()}"
    os.environ["LANGCHAIN_PROJECT"] = test_name

    retriever = Retriever(db, k=3, search_type="mmr")
    prompter = Prompter()
    qa_prompt = prompter.merge_with_template(chatbot, "qa")
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=qa_prompt)

    qa = ConversationalRetrievalChain.from_llm(lc_pipeline, retriever.base_retriever, chain_type="stuff", return_source_documents=True,
                                                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT})

    start_time = time.time()
    real_as = []

    for questions in test_queries.values():
        for question in questions["questions"]:

            result = qa({"question": question, "chat_history": []})
            answer = result["answer"].strip()
        
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")

    del lc_pipeline
    del chatbot
    chatbot = []
    lc_pipeline = []
    torch.cuda.empty_cache()
    time.sleep(5)
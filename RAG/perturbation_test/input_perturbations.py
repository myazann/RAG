import json
import time
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import huggingface_hub

import torch

from RAG.chatbots import choose_bot
from RAG.utils import get_device, get_args
from RAG.loader import FileLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter

huggingface_hub.login(new_session=False)
args = get_args()
file_name = args.document
test = args.perturb_test_type

file_loader = FileLoader()
file = file_loader.load(file_name)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
texts = text_splitter.split_documents(file)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings)
chatbots = ["LLAMA2-7B-GPTQ", "LLAMA2-13B-GPTQ", "STABLE-BELUGA-7B-GPTQ", "STABLE-BELUGA-13B-GPTQ", "CLAUDE-V1", "CLAUDE-V2"] 

with open(f"{test}.json", "r") as f:
    test_queries = json.load(f)

for bot in chatbots:

    print(bot)

    device = get_device()
    chatbot = choose_bot(device, bot, gen_params={"max_new_tokens": 512, "temperature": 0})

    test_name = f"PT_{test}_{bot}_{time.time()}"
    os.environ["LANGCHAIN_PROJECT"] = test_name

    retriever = Retriever(db, k=3, search_type="mmr")
    prompter = Prompter()
    qa_prompt = prompter.merge_with_template(chatbot, "qa")
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=qa_prompt)

    qa = ConversationalRetrievalChain.from_llm(chatbot.pipe, retriever.base_retriever, chain_type="stuff", return_source_documents=True,
                                               combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT})

    start_time = time.time()
    real_as = []

    for questions in test_queries.values():
        for question in questions["questions"]:

            result = qa({"question": question, "chat_history": []})
            answer = result["answer"].strip()
        
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")

    del chatbot
    del qa
    chatbot = []
    qa = []
    torch.cuda.empty_cache()
    time.sleep(5)
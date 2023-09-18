import json
import time
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import huggingface_hub

import torch
import numpy as np

from RAG.chatbots import choose_bot
from RAG.utils import get_args, get_NoOpChain
from RAG.loader import FileLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter

huggingface_hub.login(new_session=False)
args = get_args()
file_name = args.document
test = args.perturb_test_type

file_loader = FileLoader()
file = file_loader.load(file_name)
doc = file_loader.trim_doc(file)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
texts = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings)
#chatbots = ["LLAMA2-13B-GPTQ", "VICUNA-7B-v1.5-GPTQ", "VICUNA-13B-v1.5-GPTQ", "STABLE-BELUGA-7B-GPTQ", "STABLE-BELUGA-13B-GPTQ", "CLAUDE-V1", "CLAUDE-V2"]
chatbots = ["STABLE-BELUGA-7B-GPTQ", "STABLE-BELUGA-13B-GPTQ", "CLAUDE-V1", "CLAUDE-V2"]
with open(f"{test}.json", "r") as f:
    test_queries = json.load(f)

for bot in chatbots:

    print(bot)

    chatbot = choose_bot(model_name=bot, gen_params={"max_new_tokens": 512, "temperature": 0})
    prompter = Prompter()
    qa_prompt = prompter.merge_with_template(chatbot, "qa")
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"], template=qa_prompt)
    doc_chain = load_qa_chain(chatbot.pipe, chain_type="stuff", **{"prompt": QA_CHAIN_PROMPT})

    retriever = Retriever(db)
    max_k = retriever.find_max_k(chatbot, [page.page_content for page in texts])
    max_allowed_k = 10

    if max_k > max_allowed_k:
        k_vals = list(np.arange(3, max_allowed_k+1, 1, dtype=int))
        k_vals.extend(["max", "filtered"])
    else:
        k_vals = list(np.arange(3, max_k+1, 1, dtype=int))

    for k_val in k_vals:
       
        print(f"K: {k_val}")
        if k_val in ["max", "filtered"]:
            k = max_k
        else:
            k = k_val
        if chatbot.q_bit is None:
            test_name = f"PT_WITH_K_{k_val}_{test}_{bot}_{time.time()}"
        else:
            test_name = f"PT_WITH_K_{k_val}_{test}_{bot}_{chatbot.q_bit}_{time.time()}"
        os.environ["LANGCHAIN_PROJECT"] = test_name
        retriever.init_base_retriever(k=k)
        if k_val == "filtered":
            retriever.empty_filters()
            retriever.add_embed_filter(embeddings, similarity_threshold=0.2)
            retriever.init_comp_retriever()
            chain_retriever = retriever.comp_retriever
        else:
            chain_retriever = retriever.base_retriever

        qa = ConversationalRetrievalChain(retriever=chain_retriever, combine_docs_chain=doc_chain, 
                                          question_generator=get_NoOpChain(chatbot.pipe), return_source_documents=True)

        start_time = time.time()

        for questions in test_queries.values():
            for question in questions["questions"]:
                qa({"question": question, "chat_history": []})
            
        end_time = time.time()
        print(f"Took {end_time - start_time} secs!\n")

    del chatbot
    del qa
    chatbot = []
    qa = []
    torch.cuda.empty_cache()
    time.sleep(5)
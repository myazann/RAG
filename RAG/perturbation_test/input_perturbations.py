import json
import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate

import GPUtil
import torch

from RAG.chatbots import choose_bot
from RAG.utils import init_env
from RAG.doc_loader import DocumentLoader
from RAG.retriever import Retriever
from RAG.enums import REPO_ID
from RAG.prompter import Prompter

args, device, ls_project_name = init_env("Perturbations")

doc_name = args.document

loader = DocumentLoader(doc_name)
doc = loader.load_doc()
doc = loader.trim_doc(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
texts = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings)

with open("original_queries.json", "r") as f:
    original_queries = json.load(f)["original_queries"]


tests = ["test1", "test2"]
chatbots = [REPO_ID.VICUNA_13B_GPTQ, REPO_ID.LLAMA2_7B_GPTQ, REPO_ID.LLAMA2_13B_GPTQ, 
            REPO_ID.STABLE_BELUGA_7B_GPTQ, REPO_ID.STABLE_BELUGA_13B_GPTQ] 

for test in tests:
    with open(f"{test}.json", "r") as f:
        test_queries = json.load(f)["questions"]

    res = {}
    for bot in chatbots:

        print(bot.name)
        res[bot.name] = {
            "Questions": [],
            "Generated Answers": [],
            "Source Docs": []
        }

        chatbot = choose_bot(device, bot, gen_params={"max_new_tokens": 512, "temperature": 0})
        lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

        retriever = Retriever(db, k=3, search_type="mmr")

        prompter = Prompter()
        qa_prompt = prompter.merge_with_template(chatbot, "qa")
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=qa_prompt)

        qa = ConversationalRetrievalChain.from_llm(lc_pipeline, retriever.base_retriever, chain_type="stuff", return_source_documents=True,
                                                   combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT})

        start_time = time.time()
        real_as = []

        for i, query in enumerate(test_queries):
            all_qs = []
            all_as = []
            all_sdocs = []

            for question in query:

                result = qa({"question": question, "chat_history": []})
                answer = result["answer"].strip()
                all_qs.append(question)
                all_as.append(answer)
                source_docs = " \n".join([page.page_content for page in result["source_documents"]])
                all_sdocs.append(source_docs)

            res[bot.name]["Questions"].append(all_qs)
            res[bot.name]["Source Docs"].append(all_sdocs)
            res[bot.name]["Generated Answers"].append(all_as)
            real_as.append(original_queries[i]["A"])
            
        end_time = time.time()
        res[bot.name]["Real Answer"] = real_as
        res[bot.name]["Elapsed Time"] = end_time - start_time
        print(f"Took {end_time - start_time} secs!\n")

        del lc_pipeline
        del chatbot
        chatbot = []
        lc_pipeline = []
        torch.cuda.empty_cache()
        GPUtil.showUtilization()

    with open(f"{test}_res.json", "w") as f:
        json.dump(res, f)
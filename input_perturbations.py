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

from chatbots import choose_bot
from utils import init_env
from doc_loader import DocumentLoader
from retriever import Retriever
from enums import REPO_ID
from prompter import Prompter

args, device, ls_project_name = init_env("Perturbations")

doc_name = args.document

loader = DocumentLoader(doc_name)
doc = loader.load_doc()
doc = loader.trim_doc(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
texts = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings)

original_queries = [{"Q": "Who are the leaders of each work package in the LESSEN project?",
  "A": """Work package leadership
o WP1.1: Jan van Rijn (UL, male, junior) o WP1.2: Arianna Bisazza (RUG, female, junior) o WP2.1: Suzan Verberne (UL, female, medior) o WP2.2: Faegheh Hasibi (RUN, female, junior) o WP3.1: Fatih Turkmen (RUG, male, junior) o WP3.2: Frederik Situmeang (HvA, male, medior) o WP4: Evangelos Kanoulas (UvA, male, senior)
Dutch Research Agenda Research along routes by Consortia (NWA-ORC) 2020/21 Full proposal form IMPACT PLAN APPROACH
Page 11
o WP5: Frederik Situmeang (HvA, male, medior) o WP6: Maarten de Rijke (UvA, male, senior) o WP7: Maarten de Rijke (UvA, male, senior) and Suzan Verberne (UL, female, medior)"""
 },
 {
  "Q": "What is the LESSEN project about?",
  "A": """4.1.3 Output The LESSEN research results have the form of insights as well as state-of-the-art algorithms and methods (plus the theoretical and experimental validation of these algorithms and methods).
In particular, LESSEN foresees the following outputs in order to be able to effectuate the changes listed in Section 4.1.1 (“outcomes”):
Output 1: Insights and algorithms for compute-efficient architectures; • Output 2: Insights and methods for efficient use of existing datasets;
• Output 3: Insights and methods for transfer learning for (low-resource) domain adaptation; • Output 4: Insights and methods for generating synthetic data; 
• Output 5: Insights and explainable methods for safe and privacy-preserving utterance generation; • Output 6: Insights and methods for the design of transparent
 conversational technology; • Output 7: A unified evaluation methodology, with benchmark datasets; • Output 8: Prototypes of the developed methods."""
 },
 {
  "Q": "Which companies are involved in LESSEN?",
  "A": "Achmea, Ahold Delhaize, Albert Heijn, Bol.com, Rasa, and KPN"
 }]

queries = [
    [
        "What are names of people responsible for different work packages in LESSEN?",
        "Can you tell me the lead person for each work package in LESSEN project?",
        "LESSEN has work packages, right? Who is managing them?",
        "Who is charge of the various work packages for LESSEN?",
        "In LESSEN work packages there are leaders. Tell me them.",
        "I need know who head each work package for LESSEN. Tell me please.",
        "What is name of manager for all work package LESSEN have?",
        "Who control work packages LESSEN?",
        "List name of lead for work packages in LESSEN.",
        "WORK PACKAGE LEADERS IN LESSEN? TELL ME."],
    [
       "Can you summarize what LESSEN project is?",
       "Give me overview of LESSEN project.",
       "What is purpose of LESSEN project? Explain please.",
       "Tell me what LESSEN project is for.",
       "I want understand what LESSEN project about. Help me.",
       "Need know what LESSEN is. Give summary.",
       "What LESSEN? Explain project.",
       "Describe what LESSEN project is.",
       "Briefly explain LESSEN project.",
       "WHAT LESSEN PROJECT? TELL ME ABOUT IT."],
    [
       "What companies working on LESSEN project?",
       "Name organizations involved with LESSEN.",
       "Which businesses part of LESSEN? Tell me.",
       "What groups and corporations in LESSEN project?",
       "I want know companies in LESSEN. Tell me them.",
       "Need list of businesses in LESSEN project. Give me.",
       "Who companies working on LESSEN? List them.",
       "List corporations involved with LESSEN.",
       "What companies partnered for LESSEN project?",
       "COMPANIES IN LESSEN PROJECT? NAME THEM."],
    ]


chatbots = [REPO_ID.VICUNA_7B_GPTQ, REPO_ID.VICUNA_13B_GPTQ, REPO_ID.LLAMA2_7B_GPTQ,
            REPO_ID.LLAMA2_13B_GPTQ, REPO_ID.STABLE_BELUGA_7B_GPTQ, REPO_ID.STABLE_BELUGA_13B_GPTQ] 

def get_chain(pipeline, retriever):
    return ConversationalRetrievalChain(pipeline, retriever)

res = {}
for bot in chatbots:

    print(bot.name)
    res[bot.name] = {
        "Questions": [],
        "Generated Answers": [],
        "Source Docs": []
    }

    chatbot = choose_bot(device, bot)
    lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

    retriever = Retriever(db, k=3, search_type="mmr")
    # retriever.add_embed_filter(embeddings)
    # retriever.add_doc_compressor(lc_pipeline)

    prompter = Prompter()
    qa_prompt = prompter.merge_with_template(chatbot, "qa")
    condense_prompt = prompter.merge_with_template(chatbot, "condense")
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=qa_prompt)
    CONDENSE_PROMPT = PromptTemplate.from_template(condense_prompt)

    qa = ConversationalRetrievalChain.from_llm(lc_pipeline, retriever.base_retriever, chain_type="stuff", return_source_documents=True,
                                               combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}, 
                                               condense_question_prompt=CONDENSE_PROMPT)

    start_time = time.time()
    for i, query in enumerate(queries):
        all_qs = []
        all_as = []
        all_sdocs = []

        for question in query:

            #print(f"\nQuestion: {question}\n")
            result = qa({"question": question, "chat_history": []})
            answer = result["answer"].strip()
            #print(f"Answer: {answer}\n")
            all_qs.append(question)
            all_as.append(answer)
            source_docs = " \n".join([page.page_content for page in result["source_documents"]])
            all_sdocs.append(source_docs)

        res[bot.name]["Questions"].append(all_qs)
        res[bot.name]["Source Docs"].append(all_sdocs)
        res[bot.name]["Generated Answers"].append(all_as)
        res[bot.name]["Real Answer"] = original_queries[i]["A"]
        
    end_time = time.time()
    res[bot.name]["Elapsed Time"] = end_time - start_time
    print(f"Took {end_time - start_time} secs!\n")

    del lc_pipeline
    del chatbot
    chatbot = []
    lc_pipeline = []
    torch.cuda.empty_cache()
    GPUtil.showUtilization()

with open("perturb_res.json", "w") as f:
    json.dump(res, f)

"""
eval = [{question: 
        real_answer: 
        generated_answer:
        }]


from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

client = Client()

dataset = client.create_dataset(ls_project_name)

evaluation_config = RunEvalConfig(
    evaluators=[
        "qa",
        "context_qa",
        "cot_qa",
    ]
)

    runs = client.list_runs(
    project_name=ls_project_name,
    execution_order=1,
    error=False,
    )
    
    for run in runs:
        client.create_example(
            inputs=run.inputs,
            outputs=run.outputs,
            dataset_id=dataset.id,
        )

    run_on_dataset(
    client,
    ls_project_name,
    lc_pipeline,
    #get_chain(lc_pipeline, retriever.base_retriever),
    evaluation=evaluation_config,
    )
"""
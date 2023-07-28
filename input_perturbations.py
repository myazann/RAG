import json

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline
import GPUtil
import torch

from chatbots import choose_bot
from utils import init_env
from doc_loader import DocumentLoader
from retriever import Retriever
from enums import REPO_ID

args, device = init_env("Perturbations")
doc_name = args.document

loader = DocumentLoader(doc_name)
doc = loader.load_doc()
doc = loader.trim_doc(doc)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
texts = text_splitter.split_documents(doc)

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_documents(texts, embeddings)

# Original query: "Who are the leaders of each work package in the LESSEN project?"

queries = ["Who is responsible for each work package in the LESSEN project?",
"Can you tell me the leaders for the different work packages in LESSEN?",
"In LESSEN, what are the work packages and who is leading them?",
"For LESSEN - what are the work packages and their leaders?",
"What are the work packages in LESSEN and who is managing them?",
"LESSEN work packages - who is in charge of each one?",
"Whos leading the work packages for LESSEN?",
"Who manages the different work packages for the LESSEN project?",
"In charge work packages LESSEN?",
"LESSEN work package responsible people?",
"I would like to know the lead for each LESSEN work package please",
"Could I get the names of the LESSEN wp leaders",
"Tell me LESSEN wp and manager",
"Who head LESSEN work package?",
"Need know LESSEN work package leader",
"What person manage LESSEN work package?",
"Who run each LESSEN work package?",
"Who responsible LESSEN wp?",
"LESSEN work package lead?",
"Who lead LESSEN work package?"]

chatbots = [REPO_ID.GPT4ALL_13B_GPTQ, REPO_ID.VICUNA_7B_GPTQ, REPO_ID.VICUNA_13B_GPTQ, REPO_ID.LLAMA2_7B_GPTQ, REPO_ID.LLAMA2_13B_GPTQ] 

res = {}
for bot in chatbots:

    print(bot.name)
    res[bot.name] = {
        "Questions": [],
        "Answers": []
    }

    chatbot = choose_bot(device, bot)
    lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)

    retriever = Retriever(db, k=3, search_type="mmr")
    # retriever.add_embed_filter(embeddings)
    # retriever.add_doc_compressor(lc_pipeline)

    chat_history = []
    qa = ConversationalRetrievalChain.from_llm(lc_pipeline, retriever.base_retriever, chain_type="stuff", return_source_documents=False)

    for query in queries:
        print(f"\nQuestion: {query}\n")
        result = qa({"question": query, "chat_history": chat_history})
        answer = result["answer"].strip()
        print(f"Answer: {answer}\n")
        res[bot.name]["Questions"].append(query)
        res[bot.name]["Answers"].append(answer)

    del lc_pipeline
    del chatbot
    chatbot = []
    lc_pipeline = []
    torch.cuda.empty_cache()

    GPUtil.showUtilization()

with open("perturb_res.json", "w") as f:
    json.dump(res, f)
import json
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from utils import init_env
from chatbots import choose_bot

device = init_env("COQA")

with open("coqa/coqa-dev-v1.0.json", "rb") as f:
  coqa_dev = json.load(f)
  
out_dir = os.path.join("coqa", "dev")
os.makedirs(out_dir, exist_ok=True)
text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)

embeddings = HuggingFaceEmbeddings()

chatbot = choose_bot(device)
lc_pipeline = HuggingFacePipeline(pipeline=chatbot.pipe)


template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say unknown, don't try to make up an answer. Keep it as short as possible, you don't need to form a sentence,  use a couple of words.
Context: {context}
Question: {question}
Answer:"""

template = """[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don't know the answer to a question, please don't share false information.
<</SYS>>
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say unknown, don't try to make up an answer. Keep it as short as possible, you
don't need to form a sentence,  use a couple of words.
Context: {context}
Question: {question}[/INST]""" 

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

condense_prompt = """
I am going to give you a chat history between a human and an assistant, and a follow-up question. Replace the pronouns in the follow-up question with the corresponding noun by using the chat history. Don't replace anything except pronouns. If there are no pronouns or if the follow-up question is clear enough, do not replace anything. Same with the follow-up question, standalone question should only be one sentence, and it should basically ask the same thing only with different words.
Chat History: {chat_history}
Follow-up question: {question}
Standalone question:
"""

condense_prompt = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don't know the answer to a question, please don't share false information.
<</SYS>>
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History: {chat_history}
Follow-up question: {question}
Standalone question:[/INST]
"""

CONDENSE_PROMPT = PromptTemplate.from_template(condense_prompt)

for i, sample in enumerate(coqa_dev["data"]):

  if i == 20:
    break

  story = sample["story"]
  questions = [q["input_text"] for q in sample["questions"]]
  gt_answers = [q["input_text"] for q in sample["answers"]]
  
  filename = os.path.join(out_dir, f"{i}.txt")
  with open(filename, "w+") as f:
    f.writelines(story)
  
  loader = TextLoader(filename)
  documents = loader.load()
  
  docs = text_splitter.split_documents(documents)
  
  db = Chroma.from_documents(docs, embeddings)
  
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})

  qa = ConversationalRetrievalChain.from_llm(lc_pipeline, retriever, combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}, 
  # condense_question_prompt=CONDENSE_PROMPT
  )

  chat_history = []
  
  print(story)
  print()
  for q, gt_a in zip(questions, gt_answers):
    print(f"Question: {q}")
    #res = qa_chain({"query": q})
    
    result = qa({"question": q, "chat_history": chat_history})
    answer = result["answer"].strip()
    
    print(f"Answer: {answer}")
    chat_history.append((q, answer))
    
  
"""
     
  chat_history = []
  
  print(story)
  print()
  for q in questions:
    print(f"Question: {q}")
    #res = qa_chain({"query": q})
    
    result = qa({"question": q, "chat_history": chat_history})
    answer = result["answer"].strip()
    
    print(f"Answer: {answer}")
    chat_history.append((q, answer))
    
    
embeddings = HuggingFaceEmbeddings()

pipe = choose_bot(device)
llm = HuggingFacePipeline(pipeline=pipe)



args = get_args()
device = args.device

models = available_repos()

for v in models.values():

  model_name = v.split("/")[1].split("-")
  model_name = "-".join(model_name[:2])
  print(v)
  pipe = choose_bot(device, v)
                
  for prompt in all_prompts:
    #print(prompt)
    ans = pipe(prompt)
    print(ans[0]["generated_text"])
  
  
  
Chat History:
Human: What color was the dog
Assistant: white
Human: What was she doing
Assistant: She was running
Follow-up question: Where did she live?
Standalone question: Where did the dog live?
"""
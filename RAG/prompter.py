from RAG.utils import strip_all

class Prompter():

    def __init__(self):

        self.prompt_dict = {
            "condense": self.condense_q_prompt(),
            "qa": self.qa_prompt(),
            "eval_qa": self.eval_qa_prompt(),
            "multi_query": self.multi_query_prompt()
        }

    def stripped_prompts(self, prompt):
        return strip_all(prompt)
    
    def eval_qa_prompt(self):
        return self.stripped_prompts("""Your job is to evaluate an answer given the question and the solution. You will output a score between 0 and 100 for the following categories:
        -Correctness: How correct is the answer given the solution? The answer does not have to exactly be the same as the solution but the context should be similar and it should include most of the information given in the solution. If the answer does not mention most of the solution, give it a low score. Also, the answer should not include any false information.
        -Relevance: How relevant is the answer for the question, given the solution? You will check if the answer goes of topic and starts mentioning unrelated information to the question.
        -Coherence: Is the answer coherent? Does it repeats the same sentence over and over or starts talking about completely unrelated and illogical things? How relevant is the answer for the question (This should be a priority, give a low score to answers with irrelevant information)?
        You are a very strict evaluator and would only give a score above 75 if the answer is perfect. If the answer is not perfect but still acceptable, give a score between 50-100. If the answer does not resemble the solution and talks about irrelevant stuff, give a score below 50. Finally, you are going to give a brief explanation about the scores you gave. Your output will be as follows:
        Correctness: <correctness_score>
        Relevance: <relevance_score>
        Coherence: <coherence_score>
        Explanation: <your explanation about why you gave those score>
        Question: 
        {question}
        Solution: 
        {solution}
        Answer:
        {answer}""")
    
    def gen_sim_queries(self, test_id):

        # Prompt to generate similar questions. This prompt works best with Claude 2.0, so it may not produce 
        # desirable outputs in different chatbots.
        if test_id == 1:
            return self.stripped_prompts("""Your task is to create 10 sentences that have a very similar meaning to the question provided. The same question can be formatted very differently depending on the user, the length and the level of formality can vary, and there can be grammatical errors or typos. Sometimes, users don't even form sentences or ask clear questions, so some of your generations should resemble that. The newly created sentences should resemble sentences generally inputted to chatbots by customers, so mimic that style. Put the generated sentences in a Python list.
            Question: {question}""")
        elif test_id == 2:
            return self.stripped_prompts("""Your task is to change the spellings and change the casing of the characters for a given sentence. You also can also remove or repeat the punctutaion mark at the end of the sentecen. You will create 10 new sentences from the original one. Put the generated sentences in a Python list.                  
            Original sentence: {sentence}
            Variations:""")
        else:
            print("No such test id!")

    def qa_prompt(self):
        return self.stripped_prompts("""
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say I am sorry but I don't know the answer, don't try to make up an answer.
        {context}
        {question}
        Answer:""")

    def condense_q_prompt(self):
        return self.stripped_prompts("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        {chat_history}
        {question}
        Standalone question:""")
    
    def multi_query_prompt(self):
        return self.stripped_prompts("""You are an AI language model assistant. Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions seperated by newlines. Do not output anything besides the questions, and don't put a blank line between the questions.
        Original question: {question}""")

    def merge_with_template(self, chatbot, prompt_type):

        prompt = self.prompt_dict[prompt_type]
        template = chatbot.prompt_template()
        if template is not None:
            prompt = template.format(prompt=prompt)
        return prompt
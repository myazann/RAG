from RAG.utils import strip_all

class Prompter():

    def __init__(self):

        self.prompt_dict = {
            "condense": self.get_condense_q_prompt(),
            "qa": self.get_qa_prompt(),
            "eval_qa": self.eval_qa_prompt()
        }

    def stripped_prompts(self, prompt):
        return strip_all(prompt)
    
    def eval_qa_prompt(self):
        return self.stripped_prompts("""I want you to act as an evaluator. I will give you a question, the solution, and the prediction, and you will give a score between 0 and 100 to the prediction. You will evaluate whether the prediction is relevant to the question by using the solution as the reference. You should take into account whether the prediction goes off topic, or contains unrelated, not mentioned or false information. Factuality and mention of unrelated information should be your priority, those answers should have a low score. If the prediction does not answer the question but is still trying to be helpful or polite, give it a score of 25. Your output will be a Python dictionary with the score and explanation as keys and the score you give and a brief explanation about why you have given that score as values.                    
        Question: 
        {question}
        Solution: 
        {real_answer}
        Prediction:
        {gen_answer}""")
    
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

    def get_qa_prompt(self):
        return self.stripped_prompts("""
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say I am sorry but I don't know the answer, don't try to make up an answer.
        {context}
        {question}
        Answer:""")

    def get_condense_q_prompt(self):
        return self.stripped_prompts("""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
        {chat_history}
        {question}
        Standalone question:""")

    def merge_with_template(self, chatbot, prompt_type):

        prompt = self.prompt_dict[prompt_type]
        template = chatbot.prompt_template()
        if template is not None:
            prompt = template.format(prompt=prompt)
        return prompt
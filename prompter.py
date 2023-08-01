from utils import strip_all

class Prompter():

    def __init__(self):

        self.prompt_dict = {
            "condense": self.get_condense_q_prompt(),
            "qa": self.get_qa_prompt()
        }

    def stripped_prompts(self, prompt):
        return strip_all(prompt)
    
    def gen_sim_queries(self):

        # Prompt to generate similar questions. This prompt works best with Claude 2.0, so it may not produce 
        # desirable outputs in different chatbots.
        return self.stripped_prompts("""Your task is to create 20 sentences that have a very similar meaning to the question provided. 
        The same question can be formatted very differently depending on the user, the length and the level of formality
        can vary, and there can be grammatical errors or typos. Sometimes, users don't even form sentences or ask clear questions,
        so some of your generations should resemble that. The newly created sentences should resemble sentences generally inputted
        to chatbots by customers, so mimic that style. Put the generated sentences in a Python list.
        Question: {question}""")

    def get_qa_prompt(self):
        return self.stripped_prompts("""
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
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
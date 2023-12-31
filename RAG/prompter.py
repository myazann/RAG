class Prompter():

    def lamp_prompt(self, dataset, k=True):
        if dataset == 2:
            return """Your task is to categorize an article by choosing from one of the provided categories. You will only output the category name and nothing else.
                Article: 
                {article}
                Categories: [women, religion, politics, style & beauty, entertainment, culture & arts, sports, science & technology, travel, business, crime, education, healthy living, parents, food & drink]"""
        
        elif dataset == 3:
            if k:
                return """
                Here are a couple of review-rating pairs of a user. 
                <EXAMPLES>
                {examples}
                </EXAMPLES>
                With the given examples, give a score between [1, 2, 3, 4, 5] to the following review by the same user. Only output the score and nothing else.
                Review: 
                {prof_text}
                Score:
                """
            else:
                return """
                Give a score between [1, 2, 3, 4, 5] to the following review. Only output the score and nothing else.
                Review: 
                {prof_text}
                Score:
                """  
        elif dataset == 5:
            if k:
                return """Here are a couple of abstract-title pairs of an author:
                <EXAMPLES>
                {examples}
                </EXAMPLES>
                With the given examples, generate a title for the given abstract by the same author. Only output the title and nothing else:
                Abstract:
                {prof_text}
                Title:"""
            else:
                return """Your task is to generate a title for the given abstract. You will only output the title and nothing else.
                Abstract:
                {prof_text}
                Title:"""

    def eval_qa_prompt(self):
        return """Your job is to evaluate an answer given the question and the solution. You will output a score between 0 and 10 for the following categories:
        -Correctness: How correct is the answer given the solution? The answer does not have to exactly be the same as the solution but the context should be similar and it should include most of the information given in the solution. If the answer does not mention most of the solution, give it a low score. The answer should not include any false information.
        -Relevance: How relevant is the answer for the question, given the solution? You will check if the answer goes of topic and starts mentioning unrelated information to the question.
        -Coherence: Is the answer coherent? Does it repeats the same sentence over and over or starts talking about completely unrelated and illogical things? How relevant is the answer for the question (This should be a priority, give a low score to answers with irrelevant information)?
        You will first analyze the solution for the mentioned categories and explain what are its pros and cons, then you will output your score for the given categories. You are a very strict evaluator and would only give a score above 7.5 if the answer is perfect. If the answer is not perfect but still acceptable, give a score around 5. If the answer does not resemble the solution and talks about irrelevant stuff, give a score close to 0. Your output will be as follows:
        Analysis: <your analysis>
        Correctness: <correctness_score>
        Relevance: <relevance_score>
        Coherence: <coherence_score>
        Question: 
        {question}
        Solution: 
        {solution}
        Answer:
        {answer}"""
    
    def gen_sim_queries(self, test_id):
        # Prompt to generate similar questions. This prompt works best with Claude 2.0, so it may not produce 
        # desirable outputs in different chatbots.
        if test_id == 1:
            return """Your task is to create 10 sentences that have a very similar meaning to the question provided. The same question can be formatted very differently depending on the user, the length and the level of formality can vary, and there can be grammatical errors or typos. Sometimes, users don't even form sentences or ask clear questions, so some of your generations should resemble that. The newly created sentences should resemble sentences generally inputted to chatbots by customers, so mimic that style. Put the generated sentences in a Python list.
            Question: {question}"""
        elif test_id == 2:
            return """Your task is to change the spellings and change the casing of the characters for a given sentence. You also can also remove or repeat the punctutaion mark at the end of the sentecen. You will create 10 new sentences from the original one. Put the generated sentences in a Python list.                  
            Original sentence: {sentence}
            Variations:"""
        else:
            print("No such test id!")

    def qa_prompt(self):
        return """Your job is to give an answer to the user input at the end. You can use the following pieces of context and the chat history to give an answer. The context and chat history are to help you give a satisfactory answer to the user input, but if the user input is irrelevant to the context and history, you don't have to use them to answer it. If you don't know the answer to the input, don't try to make up an answer.
        Context:
        {context}
                                     
        Chat History:
        {chat_history}
                                     
        User Input:
        {question}"""

    def condense_q_prompt(self):
        return """Output the given summary of the conversation history and the question, as is. Do not change anything.                                     
        Summary of the conversation history:
        {chat_history}
        Question:
        {question}"""
    
    def memory_summary(self):
        return """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary. Do not output anything except the summary. Do not make a very long summary, keep it short.
        EXAMPLE
        Current summary:
        The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

        New lines of conversation:
        Human: Why do you think artificial intelligence is a force for good?
        AI: Because artificial intelligence will help humans reach their full potential.

        New summary:
        The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
        END OF EXAMPLE

        Current summary:
        {summary}

        New lines of conversation:
        {new_lines}

        New summary:"""
    
    def multi_query_prompt(self, n=3):
        return """You are an AI language model assistant. Your task is to generate {n} different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions seperated by newlines. Do not output anything besides the questions, and don't put a blank line between the questions.
        Original question: {question}""".format(n=n)
    
    def csv_prompt(self):
        return """A user wants to gain insights about a pandas dataframe. Your job is to translate the user command into Python code so that they can run the code to learn more about the dataframe. The csv is already loaded into the dataframe, and you are going to refer to the dataframe as "df". You are going to output the code inside "```". You are not going to output anything except the code. Include a print function to show the output of the code. Here is an example:
        User Input: How many rows are there?
        Output: 
        ```
        print(df.shape[0])
        ```
        Here is the chat history between you and the user:
        {chat_history}
         
        {user_input}
        Output:"""
from RAG.output_formatter import strip_all

class Prompter():

    def lamp_prompt(self, dataset, prof_text, examples=None, k=True):
        if dataset == 3:
            if k:
                return [
                    {
                        "role": "system",
                        "content": strip_all("""Your task is to give a score between [1, 2, 3, 4, 5] to a user review by utilizing the provided review-score pairs from the same user. Only output the score and nothing else.""")
                    },
                    {
                        "role": "user", 
                        "content": strip_all(f"""{examples}
                                                Review: {prof_text}
                                                Score:""")
                    }]
            else:
                return [
                    {
                        "role": "system",
                        "content": strip_all("Give a score between [1, 2, 3, 4, 5] to the following review. Only output the score and nothing else.")
                    },
                    {
                        "role": "user",
                        "content": strip_all(f"""Review: {prof_text}
                                                Score:""")
                    }]

        elif dataset == 5:
            if k:
                return [
                    {
                        "role": "system",
                        "content": strip_all("""Your task is to generate a title for the given abstract by utilizing the provided abstract-title pairs from the same author. Only output the title and nothing else.""")
                    },
                    {
                        "role": "user",
                        "content": strip_all(f"""{examples}
                                                Abstract: {prof_text}
                                                Title:""")
                    }]
            else:
                return [
                    {
                        "role": "system",
                        "content": strip_all("Your task is to generate a title for the given abstract. You will only output the title and nothing else.")

                    },
                    {
                        "role": "user",
                        "content": strip_all(f"""Abstract: {prof_text}
                                                 Title:""")
                    }]

    def eval_qa_prompt(self, question, solution, answer):
        return [
            {
                "role": "system",
                "content": strip_all("""
                Your job is to evaluate an answer given the question and the solution. You will output a score between 0 and 10 for the following categories:
                -Correctness: How correct is the answer given the solution? The answer does not have to exactly be the same as the solution but the context should be similar and it should include most of the information given in the solution. If the answer does not mention most of the solution, give it a low score. The answer should not include any false information.
                -Relevance: How relevant is the answer for the question, given the solution? You will check if the answer goes of topic and starts mentioning unrelated information to the question.
                -Coherence: Is the answer coherent? Does it repeats the same sentence over and over or starts talking about completely unrelated and illogical things? How relevant is the answer for the question (This should be a priority, give a low score to answers with irrelevant information)?
                You will first analyze the solution for the mentioned categories and explain what are its pros and cons, then you will output your score for the given categories. You are a very strict evaluator and would only give a score above 7.5 if the answer is perfect. If the answer is not perfect but still acceptable, give a score around 5. If the answer does not resemble the solution and talks about irrelevant stuff, give a score close to 0. Your output will be as follows:
                Analysis: <your analysis>
                Correctness: <correctness_score>
                Relevance: <relevance_score>
                Coherence: <coherence_score>""")
            },
            {
                "role": "user",
                "content": strip_all(f"""Question: 
                                        {question}
                                        Solution: 
                                        {solution}
                                        Answer:
                                        {answer}""")
            }]

    def qa_prompt(self, context, chat_history, question):
        return [
            {
                "role": "system",
                "content": strip_all("Your job is to give an answer to the user input at the end. You can use the pieces of context and the chat history to give an answer. The context and chat history are to help you give a satisfactory answer to the user input, but if the user input is irrelevant to the context and history, you don't have to use them to answer it. If you don't know the answer to the input, don't try to make up an answer."),
            }, 
            {
                "role": "user",
                "content": strip_all(f"""Context:
                                        {context}
                                        Chat History:
                                        {chat_history}                     
                                        User Input:
                                        {question}""")
            }]
    
    def conv_agent_prompt(self, info, chat_history, user_input):
        return [
            {
                "role": "system",
                "content": strip_all("You are an agent that has a conversation with a user. Information related to the user input is going to be provided to you during the conversation. If you think that the information is relevant to answer the user, you can use it. Sometimes, the information may be unrelated or may not contain the answer the user is looking for. For those cases, do not use the provided information. Therefore, you need to decide whether the related information is actually useful to give the user a satisfactory answer. The provided information may contradict what you know. In those cases, provided information has priority. A summary of the chat history between you and the user is also going to be included after the related information to inform you about the current state of the conversation. Your answer should be in the style of a conversational assistant. Do not mention that you have used the provided information or the chat history for your answer. If you do not know the answer, do not say that the information is not provided, state that you do not know the answer. If the user input is empty or composed of a single word that is not related to the provided information and the chat history, ask the user for clarification.")
            }, 
            {
                "role": "user",
                "content": strip_all(f"""Related information:
                                        {info}
                                        Summary of the chat history:
                                        {chat_history}
                                        User input:
                                        {user_input}""")
            }]

    def query_gen_prompt(self, summary, user_input):
        return [
            {
                "role": "system",
                "content": strip_all("""Your task is to transform user inputs into web search queries, given the summary of the chat history. User input can be ambigous or not clear. For example, the user input may reference an object from the previous interaction. For those cases, clear the ambiguity using the summary. If the input is already in the format of a query, output the user input without any modifications. Do not output anything expect the query and do not give an explanation. For the following situations, do not transform the input into a query and output only "NO QUERY": 1) If the user input is directing a question to someone using a pronoun 2)If the input is composed of a single word like a number or an object 3) If the user query is ambigous but the chat summary is empty or it does not provide information for clarification.
                                        Here are some examples:
                                        Chat Summary:
                                        User Input: Project_Proposal.pdf
                                        Query: Project_Proposal.pdf
                                        Chat Summary: The user asked the assistant what are the most common approaches for normalizing inputs. Assistant gave the user an overview.
                                        User Input: 5
                                        Query: NO QUERY
                                        Chat Summary: The user claimed that they are interested in football and asked the assistant about the origins of football. Assistant explained that modern football was originated in England in the mid 19th century.
                                        User Input: Which teams were in the finals of the last world cup
                                        Query: Last world cup finalists
                                        Chat Summary: The user is complaining about the actions of his ex-boyfriend, who stopped communicating with the user. The assistant gave an overview about possible reasons, trying to show both sides of the story. The user told the assistant that they are very angry.
                                        User Input: Why does he act like this?
                                        Query: NO QUERY
                                        Chat Summary: The user mentions that they like the video game Starcraft 2 a lot, then explains the assistant that there is a very powerful unit in the game called colossus. The user explains that the colossus is prone to falling.
                                        User Input: Why did the it fall over?
                                        Query: colossus fall reasons
                                        Chat Summary:
                                        User Input: What is the best way to do it?
                                        Query: NO QUERY
                                        Chat Summary: The user asked why "Attention is All You Need" paper is a milestone in NLP research. The assistant explained the attention concept, and gave an overview of how it changed the current paradigm in NLP.
                                        User Input: What are the findings of the paper?
                                        Query: attention is all you need findings
                                        Chat Summary:
                                        User Input: best new year resolutions
                                        Query: best new year resolutions
                                        Chat Summary: User is a big fan of the brand "Apple", and they own all their products. The assistant congratulates the user and gives examples about the popularity of Apple all around the world. The user then wants an overview of the differences between iOS and Android products. The assistant explains that they are both operating systems and provides the user with a table highlighting their differences.
                                        User Input: Apple
                                        Query: NO QUERY""")
        }, 
        {
            "role": "user",
            "content": strip_all(f"""Chat Summary: {summary}
                                    User Input: {user_input}
                                    Query:""")
        }]
    
    def memory_summary(self, summary, new_lines):
        return [
            {
                "role": "system",
                "content": strip_all("""Your task is to summarize a conversation between a user and an assistant. The current summary and the new lines in the conversation will be provided to you. Progressively summarize the lines of conversation provided, adding onto the previous summary. The new summary should include key information that may come up later in the conversation, such as what the user asked and what did the user and the assistant talked previously. If there is no current summary, only use the new lines of conversation for the summary. Do not output anything except the summary, and do not make it very long, keep it short. It should not exceed 256 words.""")
            }, 
            {
                "role": "user",
                "content": strip_all(f"""Current summary:
                                        {summary}
                                        New lines of conversation:
                                        {new_lines}
                                        New summary:""")
            }]
      
    def multi_query_prompt(self, question, n=3):
        return [
            {
                "role": "system",
                "content": strip_all(f"Your task is to generate {n} different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions seperated by newlines. Do not output anything besides the questions, and don't put a blank line between the questions.")
            }, 
            {
                "role": "user",
                "content": strip_all(f"Question: {question}")
            }]
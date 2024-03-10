from RAG.output_formatter import strip_all

class Prompter():

    def lamp_prompt(self, dataset, prof_text, examples=None):
        if dataset == 3:
            if examples:
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
            if examples:
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
    
    def conv_agent_prompt(self, info, user_input):
        return [
            {
                "role": "system",
                "content": strip_all("You are an agent that has a conversation with a user. Information related to the user input is going to be provided to you during the conversation. If you think that the information is relevant to answer the user, you can use it. Sometimes, the information may be unrelated or may not contain the answer the user is looking for. For those cases, do not use the provided information. Therefore, you need to decide whether the related information is actually useful to give the user a satisfactory answer. The provided information may contradict what you know. In those cases, provided information has priority. Do not mention that you have used the provided information. If you do not know the answer, do not say that the information is not provided, state that you do not know the answer. If the user input is empty or composed of a single word that is not related to the provided information, ask the user for clarification. Your answer should be in the style of a conversational assistant.")
            }, 
            {
                "role": "user",
                "content": strip_all(f"""Related information:
                                        {info}
                                        User input:
                                        {user_input}""")
            }]

    """
                                            Here are some examples:

                                            Chat History:
                                            User: What are the most common approaches for normalizing inputs?
                                            Assistant: The most common approaches for normalizing inputs are:
                                            1. Min-Max Scaling: Scales the input features to a fixed range, usually between 0 and 1.
                                            2. Z-Score Normalization: Transforms the input features to have zero mean and unit variance.
                                            3. Decimal Scaling: Moves the decimal point of the input features to normalize their range.
                                            4. Log Scaling: Applies logarithm to the input features to reduce the effect of outliers and skewness.
                                            5. Robust Scaling: Uses median and interquartile range to scale the input features, making it robust to outliers.
                                            These techniques help to bring the input features to a similar scale, which can improve the performance and convergence of machine learning models.
                                            User: 5
                                            Query: NO QUERY

                                            Chat History:
                                            User: I'm interested in football. Can you tell me about the origins of football?
                                            Assistant: Modern football originated in England during the mid-19th century. The Football Association (FA) was formed in 1863, establishing the first official rules of the game. These rules laid the foundation for the sport we know today. The game quickly gained popularity and spread to other parts of the United Kingdom and eventually to the rest of the world. The first international football match was played in 1872 between England and Scotland. As the sport grew, more national associations were formed, and international competitions like the FIFA World Cup (first held in 1930) and the UEFA European Championship (first held in 1960) were established. Today, football is the most popular sport in the world, with millions of players and fans across the globe.
                                            User: Which teams were in the finals of the last world cup
                                            Query: Last football world cup finalists

                                            Chat History:
                                            User: I can't believe my ex-boyfriend stopped talking to me. It's been weeks, and he won't respond to any of my messages.
                                            Assistant: It's understandable that you're feeling hurt and frustrated by your ex-boyfriend's actions. There could be several reasons why he has stopped communicating with you:
                                            1. He might need some space and time to process the breakup and his emotions.
                                            2. He could be trying to move on and believes that cutting off communication is the best way to do so.
                                            3. There might be some unresolved issues or feelings of hurt from your relationship that make it difficult for him to maintain contact.
                                            4. He may be focusing on other aspects of his life, such as work, friends, or personal growth.
                                            It's important to remember that you can't control his actions, and it's best to focus on your own healing and well-being. Give yourself time to grieve the loss of the relationship and surround yourself with supportive friends and family. If his lack of communication persists and you find it difficult to cope, consider seeking the help of a therapist or counselor.
                                            User: I'm just so angry. Why does he act like this?
                                            Query: NO QUERY

                                            Chat History:
                                            User: I love playing Starcraft 2. It's such an amazing game!
                                            Assistant: That's great! Starcraft 2 is indeed a popular and engaging real-time strategy game. What do you enjoy most about it?
                                            User: I really like the different units and their abilities. There's this one unit called the Colossus that's super powerful, but it has a funny weakness.
                                            Assistant: Ah, yes, the Colossus! It's a massive Protoss robotic unit that can deal devastating damage to ground units with its long-range thermal lances. However, you mentioned it has a weakness. Can you elaborate on that?
                                            User: Well, despite its size and power, the Colossus is actually quite prone to falling over! It has these thin, spindly legs that don't seem to provide the best stability.
                                            Assistant: You're right! The Colossus's long, thin legs make it vulnerable to toppling over, especially when it's attacked or trying to navigate uneven terrain. This can be a significant drawback in battle, as a fallen Colossus is much easier to destroy and can't attack until it gets back up. It's an interesting design choice that balances the unit's immense firepower with a unique weakness.
                                            User: Why did it fall over?
                                            Query: starcraft 2 colossus fall reasons

                                            Chat History:
                                            User: What is the best way to do it?
                                            Query: NO QUERY

                                            Chat History:
                                            User: Why is the "Attention is All You Need" paper considered a milestone in NLP research?
                                            Assistant: The "Attention is All You Need" paper, published in 2017 by Vaswani et al., is considered a milestone in Natural Language Processing (NLP) research because it introduced the Transformer architecture, which revolutionized the field. The key ideas from the paper are:

                                            1. Self-Attention Mechanism: The paper introduced the self-attention mechanism, which allows the model to weigh the importance of different parts of the input sequence when processing each word. This enables the model to capture long-range dependencies and contextual information more effectively than previous architectures, such as Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs).

                                            2. Transformer Architecture: The Transformer is a novel neural network architecture that consists of an encoder and a decoder, both of which are composed of multiple layers of self-attention and feed-forward neural networks. This architecture is highly parallelizable and can be trained more efficiently than RNNs.

                                            3. Positional Encoding: Since the Transformer does not have an inherent notion of word order, the paper introduced positional encoding to inject information about the relative position of words in the input sequence.

                                            The Transformer architecture has become the foundation for state-of-the-art models in various NLP tasks, such as machine translation, text summarization, and question answering. It has also inspired the development of powerful language models like BERT, GPT, and their variants, which have pushed the boundaries of NLP even further.
                                            User: What are the findings of the paper?
                                            Query: attention is all you need findings

                                            Chat History:
                                            User: best new year resolutions
                                            Query: best new year resolutions

                                            Chat History:
                                            User: I'm a huge fan of Apple products. I have an iPhone, iPad, MacBook, and even an Apple Watch!
                                            Assistant: That's impressive! Apple has certainly built a loyal customer base with their innovative products and seamless ecosystem. It's no wonder that Apple is one of the most valuable companies in the world. Many people appreciate the sleek design, user-friendly interface, and high-quality features of Apple devices. The integration between Apple products, such as the ability to sync data and easily share files, is also a significant draw for many users. Apple's popularity spans the globe, with a strong presence in countries like the United States, United Kingdom, Japan, and China.
                                            User: I've been considering switching to an Android phone, but I'm not sure about the differences between iOS and Android. Can you give me an overview?
                                            Assistant: Sure! iOS and Android are the two most popular mobile operating systems. Here's a table highlighting some key differences between them:

                                            | Feature | iOS | Android |
                                            |---------|-----|---------|
                                            | Manufacturer | Apple | Various (e.g., Samsung, Google, LG) |
                                            | Customization | Limited | Highly customizable |
                                            | App Store | Apple App Store | Google Play Store |
                                            | Voice Assistant | Siri | Google Assistant |
                                            | Software Updates | Consistent and timely | Varies by manufacturer and model |
                                            | Hardware Options | Limited to Apple devices | Wide range of devices from different manufacturers |
                                            | Default Apps | Cannot be changed | Can be changed |
                                            | File Management | Limited file system access | More accessible file system |

                                            Both iOS and Android have their strengths and weaknesses. iOS is known for its simplicity, security, and seamless integration with other Apple products. Android, on the other hand, offers more flexibility, a wider range of hardware options, and a more open ecosystem. Ultimately, the choice between iOS and Android comes down to personal preference and which features matter most to you.
                                            User: Apple
                                            Query: NO QUERY
    """
    def query_gen_prompt(self, chat_history, user_input):
        return [
            {
                "role": "system",
                "content": strip_all("""Your task is to transform user messages into web search queries, given the chat history. User message can be ambiguous or not clear. For example, the user message may reference an object from the previous interaction. For those cases, clear the ambiguity using the chat history. If the message is already in the format of a query, output the user message without any modifications. Do not output anything except the query and do not give an explanation. For the following situations, do not transform the message into a query and output only "NO QUERY": 1) If the user is asking a question directly to the assistant 2) If the input is composed of a single word like a number or an object 3) If the user query is ambiguous but the chat history is empty or it does not provide information for clarification.
                                     """)
        }, 
        {
            "role": "user",
            "content": strip_all(f"""Chat History: {chat_history}
                                     User: {user_input}
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
                "content": strip_all(f"Your task is to generate {n} different search queries from the given user question to retrieve relevant documents. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions seperated by newlines. Do not output anything besides the queries, do not put the queries inside quotes, and don't put a blank line between the questions.")
            }, 
            {
                "role": "user",
                "content": strip_all(f"Question: {question}")
            }]
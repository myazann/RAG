from RAG.output_formatter import strip_all

class Prompter:
    
    def amazon_kg_construct(self, cust_hist):
        return [
            {"role": "system", "content": strip_all("""Your task is to construct a knowledge graph from this purchase history. The goal of the graph is to better understand the customer in order to make improved product recommendations for them in the future.
                                                      The graph should include the following types of nodes:
                                                      - Products
                                                      - Customer
                                                      
                                                      The graph should include the following types of relationships:
                                                      - Relationships between the customer and products indicating purchase events and the customer's sentiment towards the product (based on their rating and review)
                                                      - Relationships between products and their attributes
                                                      - Relationships between the customer and their attributes
                                                      
                                                      When extracting personal information about the customer from the reviews, do not include the full text of the reviews. Instead, identify the most salient points that capture important customer attributes and preferences.
                                                      Construct the knowledge graph and output the nodes and relationships using the following format:
                                                      Nodes
                                                      '<entity1>'
                                                      '<entity2>'
                                                      ...
                                                      Relationships
                                                      '<entity1>' -[:RELATIONSHIP_TYPE]-> '<entity2>'
                                                      '<entity1>' -[:RELATIONSHIP_TYPE {attribute: value}]-> '<entity2>'
                                                      ...
                                                      
                                                      Some additional notes:
                                                      - Aim to keep each node name concise while still being descriptive.
                                                      - For relationships, include key attributes or details in brackets when relevant.
                                                      - Ensure that all relationships are directed, specifying the start and end node.
                                                      - Do not output anything besides the nodes and the relationship.
                                                      - Capture the key characteristics of the customer such as personal information.
                                                      Analyze the customer's purchase history carefully to construct a knowledge graph that provides a comprehensive view of the customer and their purchasing behavior and preferences. The graph should enable a better understanding of the customer to drive more relevant product recommendations.""")},
            {"role": "user", "content": strip_all(f"Here is the customer purchase history:\n\n{cust_hist}")}
        ]
    
    def amazon_cust_analysis_prompt(self, cust_hist):
        return [
            {"role": "system", "content": strip_all("""Given their previous purchases, summarize the characteristics and personality of the customer in 5 points, each starting with a hyphen and separated by a new line character. Do not mention the names of the products. Only output the summary and nothing else.""")},
            {"role": "user", "content": strip_all(f"Customer Purchase History:\n\n{cust_hist}")}
        ]
    
    def amazon_np_pred_with_conv(self, cust_hist):
        return [
            {"role": "system", "content": strip_all("""Generate a conversation between a customer and a digital assistant where the assistant will look into the customer messages and their purchase history to come up with a recommendation for the customer. The assistant will ask questions to the customer to better predict what type of product they would like. While creating the customer messages, use their product reviews to replicate their conversation style and keep those messages short. Start the assistant messages with "A:" and customer messages with "C:". The assistant should describe the type of product the customer would like, instead of directly recommending a product. After the conversation ends, give a less than 10-word brief description of the product assistant recommended, as JSON.""")},
            {"role": "user", "content": strip_all(f"Customer Purchase History:\n\n{cust_hist}")}
        ]
    
    def amazon_np_pred_with_conv_claude(self, cust_hist, n_turns=5):
        return [
            {"role": "system", "content": strip_all(f"""Carefully analyze the customer's purchase history to understand their product preferences and conversation style. Pay attention to the products they have purchased, their reviews, and the language they use.
                                                        Start a conversation with the customer, asking questions to better understand what type of product they are looking for. Keep your messages concise, around 2-3 sentences each. Start each of your messages with "A:".
                                                        Generate the customer's responses based on the conversation style you observed in their purchase history. Keep their messages short and start each one with "C:".
                                                        Limit the conversation to {n_turns} turns.
                                                        After the conversation ends, provide a brief (less than 10 words) description of the type of product you would recommend for the customer based on the conversation and their purchase history. Format this description as JSON, like this:
                                                        <product_description>
                                                        {{"product_type": "your description here"}}
                                                        </product_description>
                                                        Remember, do not directly recommend a specific product. Instead, describe the type of product that would best suit the customer's needs and preferences.""")},
            {"role": "user", "content": strip_all(f"""Here is the customer's purchase history:\n<purchase_history>\n{cust_hist}</purchase_history>""")}
        ]
    
    def amazon_review_gen(self, cust_hist, prod_name, rating):
        return [
            {"role": "system", "content": strip_all(f"""Please carefully analyze the customer's previous reviews in their purchase history to understand their preferences, opinions, and writing style.
                                                        Then, imagine you are this customer and write a review for {prod_name} in a style that sounds natural coming from them. The review should align with the {rating} star rating they provided.
                                                        Write out the full text of the review the customer would write for this product. Do not explain your reasoning or predict what you think the star rating should be. Simply output the generated customer review text inside <Review> tags, like this:
                                                        <Review>
                                                        Generated review text here
                                                        </Review>""")},
            {"role": "user", "content": strip_all(f"""Here is the customer's purchase history:\n<purchase_history>\n{cust_hist}</purchase_history>\nThe customer purchased the following product:\n<product_name>\n{prod_name}\n</product_name>\nThey gave this product a rating of {rating} out of 5 stars.""")}
        ]
    
    def amazon_np_pred_with_conv_claude_cot(self, cust_hist, n=5):
        return [
            {"role": "system", "content": strip_all(f"""First, carefully analyze the customer's purchase history to understand their preferences and conversation style. Pay attention to the products they have purchased, their reviews, and any patterns in their buying behavior.
                                                        Next, generate a conversation between the customer and a digital assistant. The assistant should ask questions to better predict the type of product the customer would like. Keep the following in mind:
                                                        Start the assistant's messages with "A:" and the customer's messages with "C:".
                                                        -Generate each message in a new line.
                                                        -Keep the messages concise, around 2-3 sentences each.
                                                        -Use the customer's product reviews to replicate their conversation style in their messages.
                                                        -Focus on describing the type of product the customer would like, rather than directly recommending a specific product.
                                                        -Limit the conversation to {n} turns.
                                                        -After the conversation ends, provide a brief description of the product type the assistant recommended, using less than 10 words.
                                                        Write the conversation and the description inside tags.
                                                        Use the Chain-of-Thought style to generate the conversation, showing your reasoning and thought process.
                                                        Your output should be in the following format:
                                                        <Chain-of-Thought Reasoning>
                                                        </Chain-of-Thought Reasoning>
                                                        <conversation>
                                                        </conversation>
                                                        <description>
                                                        </description>""")},
            {"role": "user", "content": strip_all(f"""Here is the customer's purchase history:\n<purchase_history>\n{cust_hist}</purchase_history>""")}
        ]
    
    def eval_qa_prompt(self, question, solution, answer):
        return [
            {"role": "system", "content": strip_all("""Your job is to evaluate an answer given the question and the solution. You will output a score between 0 and 10 for the following categories:
                                                      -Correctness: How correct is the answer given the solution? The answer does not have to exactly be the same as the solution but the context should be similar and it should include most of the information given in the solution. If the answer does not mention most of the solution, give it a low score. The answer should not include any false information.
                                                      -Relevance: How relevant is the answer for the question, given the solution? You will check if the answer goes of topic and starts mentioning unrelated information to the question.
                                                      -Coherence: Is the answer coherent? Does it repeats the same sentence over and over or starts talking about completely unrelated and illogical things? How relevant is the answer for the question (This should be a priority, give a low score to answers with irrelevant information)?
                                                      You will first analyze the solution for the mentioned categories and explain what are its pros and cons, then you will output your score for the given categories. You are a very strict evaluator and would only give a score above 7.5 if the answer is perfect. If the answer is not perfect but still acceptable, give a score around 5. If the answer does not resemble the solution and talks about irrelevant stuff, give a score close to 0. Your output will be as follows:
                                                      Analysis: <your analysis>
                                                      Correctness: <correctness_score>
                                                      Relevance: <relevance_score>
                                                      Coherence: <coherence_score>""")},
            {"role": "user", "content": strip_all(f"""Question: 
                                                      {question}
                                                      Solution: 
                                                      {solution}
                                                      Answer:
                                                      {answer}""")}
        ]
    
    def qa_prompt(self, context, chat_history, question):
        return [
            {"role": "system", "content": strip_all("Your job is to give an answer to the user input at the end. You can use the pieces of context and the chat history to give an answer. The context and chat history are to help you give a satisfactory answer to the user input, but if the user input is irrelevant to the context and history, you don't have to use them to answer it. If you don't know the answer to the input, don't try to make up an answer.")},
            {"role": "user", "content": strip_all(f"""Context:
                                                      {context}
                                                      Chat History:
                                                      {chat_history}                     
                                                      User Input:
                                                      {question}""")}
        ]
    
    def conv_agent_prompt(self, query, context):
        return [
            {"role": "system", "content": strip_all("You are a friendly conversational agent. Your task is to help users with their questions. Information related to the user input is going to be provided to you during the conversation to help you give more plausible answers. Do not mention that you used the provided information in your output. Sometimes, the information may be unrelated or may not contain the answer the user is looking for. For those cases, you do not have to use the provided information. If you do not know the answer, or if the user question is ambigous, ask the user for clarification.")},
            {"role": "user", "content": strip_all(f"""Related information:
                                                      {context}
                                                      User input:
                                                      {query}""")}
        ]
    
    def query_gen_prompt_claude(self, user_input):
        return [
            {"role": "system", "content": strip_all("""Your task is to transform the user message into a web search query. Follow these steps:
                                                      1. If the user message is ambiguous or unclear because it references something from the earlier messages, use the chat history to resolve the ambiguity and make the query clear and specific.
                                                      2. However, if the user message is not connected to the earlier chat history, do not try to use the history to modify it. And if the user message is already in the format of a clear search query, output it as-is without any changes.
                                                      3. There are a few situations where you should output only "NO QUERY" instead of transforming the message into a search query:
                                                      - If the user is asking a question about you
                                                      - If the user message consists of only a single common word like a number or object name
                                                      4. Output ONLY the final search query. Do not provide any additional explanation or commentary.""")},
            {"role": "user", "content": strip_all(f"""Here is the most recent user message:\n<user_message>\n{user_input}\n</user_message>""")}
        ]
    
    def query_gen_prompt(self, chat_history, user_input):
        return [
            {"role": "system", "content": strip_all("""Your task is to transform user messages into web search queries, given the chat history. The user message can be be ambigous and not clear due to referencing an object from the previous messages. For those cases, clear the ambiguity using the chat history. If the user message is unconnected to the chat history, you do not have to utilize the history to clear the ambiguity. If the user message is already in the format of a query, output the user message without any modifications. Do not output anything except the query and do not give an explanation. For the following situations, do not transform the message into a query and output only "NO QUERY": 1) If the user is asking a question directly to the assistant 2) If the input is composed of a single word like a number or an object 3) If the user query is ambiguous but the chat history is empty or it does not provide information for clarification.""")},
            {"role": "user", "content": strip_all(f"""Chat History: {chat_history}
                                                      User: {user_input}
                                                      Query:""")}
        ]
    
    def memory_summary(self, summary, new_lines):
        return [
            {"role": "system", "content": strip_all("""Your task is to summarize a conversation between a user and an assistant. The current summary and the new lines in the conversation will be provided to you. Progressively summarize the lines of conversation provided, adding onto the previous summary. The new summary should include key information that may come up later in the conversation, such as what the user asked and what did the user and the assistant talked previously. If there is no current summary, only use the new lines of conversation for the summary. Do not output anything except the summary, and do not make it very long, keep it short. It should not exceed 256 words.""")},
            {"role": "user", "content": strip_all(f"""Current summary:
                                                      {summary}
                                                      New lines of conversation:
                                                      {new_lines}
                                                      New summary:""")}
        ]
    
    def multi_query_prompt(self, question, n=3):
        return [
            {"role": "system", "content": strip_all(f"Your task is to generate {n} different search queries from the given user question to retrieve relevant documents. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions seperated by newlines. Do not output anything besides the queries, do not put the queries inside quotes, and don't put a blank line between the questions.")},
            {"role": "user", "content": strip_all(f"Question: {question}")}
        ]

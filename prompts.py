# Prompt to generate similar questions. This prompt works best with Claude 2.0, so it may not produce 
# desirable outputs in different chatbots.
GEN_SIM_QUERIES = f"""I am designing questions for a question-answering chatbot. Your task is to create 20 sentences that have a very similar meaning to the question provided. 
The same question can be formatted very differently depending on the user, the length can vary, the level of formality can vary, there can be grammatical errors
or typos. Sometimes, users don't even form sentences or ask clear questions, so some of your generations should resemble that. 
The newly created sentences should resemble sentences generally inputted to chatbots by customers, so mimic that style. Provide the sentences in a Python list. 
Question: """
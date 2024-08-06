import warnings

from RAG.chatbot import choose_bot
from RAG.utils import get_args
from RAG.orchestrator import Orchestrator

def main():
    warnings.filterwarnings("ignore")

    args = get_args()
    chatbot = choose_bot()
    helper_llm = choose_bot(model_name="LLAMA-3.1-70B-PPLX")
    orchestrator = Orchestrator(chatbot, helper_llm=helper_llm)

    print("\nHello! How may I assist you? \nPress 0 if you want to quit!\nPress -1 if you want to switch the chatbot!\nPress 1 if you want to clear chat history!\nIf you want to provide a document or a webpage to the chatbot, please only input the path to the file or the url without any other text!\n")
    chat_history = []

    while True:
        query = input("User: ").strip()

        if query == "0":
            print("Bye!")
            break
        elif query == "-1":
            chatbot = choose_bot()
            continue
        elif query == "1":
            chat_history = []
            print("History cleared!")
            continue

        response = orchestrator.handle_query(query, chat_history, args)
        chat_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ])

if __name__ == "__main__":
    main()
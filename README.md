# A Framework for Retrieval Augmented Generation

A simple implementation of retrieval augmented generation for a variety of models. Currently supports retrieving from a single source which can be a URL, a PDF, a text document, or a CSV file.

## Usage

```bash
python run.py -ws 
```

- `--ws`: If passed, does web search on the queries
  
The script will ask you to choose a chatbot, and retrieval will be done from the given document or URL using Langchain. If you want to use OpenAI or Anthropic models, you need to add the corresponding API keys as environment variables. Opensource models can require a Huggingface API key.

## Installation

Run the following command after you clone the repo:

```bash
pip install -e .
```

## Todo 

Relevance | Task | Done? |
---|---| :---: |
❗❗ | Make combinations of RAG-evaluation questions | ⬜️
❗❗ | Conceptualize the flaws in evaluation | ⬜️
❗❗ | Address chunking limitations | ⬜️
❗ | Make the QA prompt more agent-like (tool usage) | ⬜️
❗ | Add weight to the base URL | ⬜️
❗ | Remove langchain dependency from the loader | ⬜️
❗ | Create a logger | ⬜️

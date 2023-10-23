# A Framework for Retrieval Augmented Generation

A simple implementation of retrieval augmented generation for a variety of models. Currently supports retrieving from a single source which can be a URL, a PDF, a text document, or a CSV file.

## Usage

```bash
python run.py -doc "URL or path to the document"
```
The script will ask you to choose a chatbot, and retrieval will be done from the given document or URL using Langchain. If you want to use OpenAI or Anthropic models, you need to add the corresponding API keys as environment variables.

## TODO 
  * Figure out a way to run AWQ 
  * Reduce the amount of dependency on Langchain 
  * Check better open-source alternatives to LLaMA 
  * Add option to pass arguments for prompts
  * Rethink the logic of counting tokens and get max k 
  * Move the lamp dataset function to lamp utils 
  * Remove langchain dependency from loader 
  * Redesign the retriever class 

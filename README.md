# A Framework for Retrieval Augmented Generation

A simple implementation of retrieval augmented generation for a variety of models. Currently supports retrieving from a single source which can be a URL, a PDF, a text document, or a CSV file.

## Usage

```bash
python run.py -doc "URL or path to the document"
```
The script will ask you to choose a chatbot, and retrieval will be done from the given document or URL using Langchain. If you want to use OpenAI or Anthropic models, you need to add the corresponding API keys as environment variables. Opensource models can require a Huggingface API key.

## Installation

Run the following command after you clone the repo:

```bash
pip install -e .
```

## LaMP Experiments

To run an experiment for the LaMP benchmark, run the following command with arguments:

```bash
python LAMP/run_exp.py -dn 5 -ds train_dev -q None -b 5 -k 3 -r bm25 -mcl 4096
```

- `--dn`: Number of the dataset, which should be an integer between 1-7.
- `--ds`: Split of the dataset that can be used. Should be between [train, dev, test]. Multiple splits can be combined by "_".
- `--q`: Type of quantization to be used. Should be from one of the following: [None, GGUF, AWQ]. None refers to not quantizing.
- `--b`: Number of quantization bits. Should be an integer between 2-8. Only relevant if GGUF quantization is used.
- `--k`: Number of retrieved documents. Can be an integer, "max", or "<i>k</i>__skip_<i>_k</i>". "0" means no retrieval, "max" means the maximum number of documents that can be put into the context window of the window, and "<i>k</i>__skip_<i>_k</i>" means skip the second k number of docs and then get the first "k" documents. This is useful to test the performance without the top retrieved documents.
- `--r`: Retriever, can be ["bm25", "contriever", "dpr"]. If k is 0, the retriever is not used.
- `--mcl`: Maximum context length to use when k=max.

The following script evaluates all the completed experiments with the validation set and produces a dataframe with Rouge scores:

```bash
python lamp_eval.py
```

The prompts used can be found in the _prompter.py_ script.

## Todo 

Relevance | Task | Done? |
---|---| :---: |
❗❗ | Make combinations of RAG-evaluation questions | ⬜️
❗❗ | Conceptualize the flaws in evaluation | ⬜️
❗ | Reduce the amount of dependency on Langchain | ⬜️
❗ | Remove langchain dependency from the loader | ⬜️
❗❗ | Rethink the logic of counting tokens and get max k | ✅
❗❗ | Figure out a way to run AWQ | ✅
❗❗ | Check better open-source alternatives to LLaMA | ✅
❗ | Redesign the retriever class | ✅
❗ | Add option to pass arguments for prompts | ✅
❗ | Move the lamp dataset function to lamp utils | ✅

## Notes

- When making evaluation questions, include prompts with multiple questions.
- Study benchmark datasets and their shortcomings

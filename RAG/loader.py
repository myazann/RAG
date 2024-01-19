import re
import requests
import os

import validators
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from git import Repo
from googlesearch import search

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader, TelegramChatFileLoader, SeleniumURLLoader, GitLoader
from langchain_community.utilities import SQLDatabase

class FileLoader():

    def __init__(self, splitter_params={"chunk_size": 2000, "chunk_overlap": 500}):
        self.splitter = RecursiveCharacterTextSplitter(**splitter_params)
    
    def load(self, file_name, web_search, pdf_loader="unstructured"):
        file_type = self.get_file_type(file_name)
        if file_type == "string" and web_search:
            print("Searching Web!")
            file_name = self.web_search(file_name)
            file_type = "url"
        if file_type == "url":
            if isinstance(file_name, str):
                if not validators.url(file_name):
                    print("Not a valid url!")
                    return None
                print("Reading web page!")
                all_urls = set()
                all_urls.add(file_name)
                all_urls = list(self.extend_url(all_urls, file_name))
            elif isinstance(file_name, list):
                all_urls = file_name
            loader = SeleniumURLLoader(urls=all_urls)
            file_name = all_urls
        elif file_type == "git":
            if not validators.url(file_name):
                print("Could not find the repository!")
                return None
            print("Reading Git repo!")
            repo_name = "/".join(file_name.split("/")[-2:])
            repo_path = f"./files/git/{repo_name}"
            if os.path.exists(repo_path):
                r = Repo(repo_path)
            else:
                r = Repo.clone_from(file_name, repo_path)
            loader = GitLoader(repo_path=repo_path, branch=r.heads[0])
        else:
            if not os.path.exists(file_name):
                print("Could not find the file!")
                return None
            print("Processing file!")
            if file_type == "db":
                print("Reading database!")
                doc = SQLDatabase.from_uri(f"sqlite:///{file_name}") 
            elif file_type == "csv":
                print("Reading CSV!")
                doc = pd.read_csv(file_name)
            elif file_type == "pdf":
                loader = self.pdf_loaders()[pdf_loader](file_name)
            elif file_type == "txt":
                loader = TextLoader(file_name)
        doc = loader.load()
        print("Done!")
        return doc, file_type

    def extend_url(self, all_urls, url):
        parsed = urlparse(url)
        base_url_path = f"{parsed.scheme}://{parsed.netloc}"
        l1_all_urls = self.get_all_links(url, base_url_path)                
        if l1_all_urls:
            all_urls.update(l1_all_urls)
            for url in l1_all_urls:
                l2_urls = self.get_all_links(url, base_url_path)
                all_urls.update(l2_urls)
        return all_urls
    
    def get_file_type(self, file_name):
        if file_name.startswith("http"):
            if "github.com" in file_name:
                return "git"
            else:
                return "url"
        else:
            file_ext = file_name.split(".")[-1]
            if len(file_ext) == 3:
                return file_ext
            else:
                return "string"

    def pdf_loaders(self):
        return {
            "structured": PyPDFLoader,
            "unstructured": UnstructuredPDFLoader,
        }

    def get_all_links(self, url, base_url):
        reqs = requests.get(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        urls = set()
        for a in soup.select("a"):
            href = a.get("href")
            if href is None:
                continue
            if base_url in href:
                urls.add(href)
            elif href.startswith("/"):
                abs_ref = f"{base_url}{href}"
                urls.add(abs_ref)
        return list(urls)
    
    def remove_empty_space(self, doc):
        for page in doc:    
            page.page_content = re.sub(r'\n+', '\n', page.page_content) 
            page.page_content = re.sub(r'\s{2,}', ' ', page.page_content)
        return doc
    
    def web_search(self, query):
        return list(search(query, tld="co.in", num=10, stop=10, pause=2))
    
    def get_processed_texts(self, file):
        all_pages = self.splitter.split_documents(self.remove_empty_space(file))
        return [page.page_content for page in all_pages], [page.metadata["source"] for page in all_pages]
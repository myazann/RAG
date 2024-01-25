import re
import requests
import os
import time

import validators
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from git import Repo
from googlesearch import search

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader, SeleniumURLLoader, GitLoader

class FileLoader():

    def __init__(self, splitter_params={"chunk_size": 2000, "chunk_overlap": 500}):
        self.splitter = RecursiveCharacterTextSplitter(**splitter_params)
    
    def load(self, file_name, web_search=False, pdf_loader="unstructured"):
        all_docs = []
        file_type = self.get_file_type(file_name)
        if file_type == "string" and web_search:
            print("Searching Web!")
            file_name = self.web_search(file_name)
            for file in file_name:
                if self.get_file_type(file) == "pdf":
                    search_doc = self.load(file)
                    all_docs.append(search_doc[0])
                    os.remove(file)
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
            if file_type == "pdf":
                loader = self.pdf_loaders()[pdf_loader](file_name)
            elif file_type == "txt":
                loader = TextLoader(file_name)
        all_docs.append(loader.load())
        print("Done!")
        return all_docs

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
        init_files = os.listdir()
        search_res = list(search(query, tld="co.in", num=10, stop=10, pause=2))
        time.sleep(5)
        new_files = os.listdir()
        diff_files = [f for f in new_files if f not in init_files]
        if diff_files:
            for f in diff_files:
                if f.endswith("pdf"):
                    search_res.extend(f)
        return search_res
    
    def get_processed_texts(self, file):
        all_chunks = []
        all_sources = []
        for f in file:
            all_pages = self.splitter.split_documents(self.remove_empty_space(f))
            for page in all_pages:
                all_chunks.append(page.page_content)
                all_sources.append(page.metadata["source"])
        return all_chunks, all_sources
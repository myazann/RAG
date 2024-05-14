import re
import requests
import os

import validators
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from git import Repo

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader, TextLoader, GitLoader

class FileLoader():

    def __init__(self, splitter_params={"chunk_size": 2000, "chunk_overlap": 500}):
        self.splitter_params = splitter_params
        self.splitter = RecursiveCharacterTextSplitter(**self.splitter_params)
    
    def load(self, file_name, pdf_loader="unstructured"):
        if isinstance(file_name, list):
            file_type = "url"
        else:
            file_type = self.get_file_type(file_name)
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
                web_pages = []
                metadatas = []
                for url in all_urls:
                    try:
                        response = requests.head(url)
                        content_type = response.headers.get('Content-Type')
                        if content_type != "application/pdf":
                            response = requests.get(url, timeout=3)
                            if response.status_code == 200:
                                html_content = response.text
                                soup = BeautifulSoup(html_content, 'html.parser')
                                page_text = soup.get_text()
                                web_pages.append(page_text)
                                metadatas.append({"source": url})
                    except Exception as e:
                        print(url)
                        print(e)
                return self.splitter.create_documents(web_pages, metadatas=metadatas)
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
        return loader.load()

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
            if file_ext in ["pdf", "txt"]:
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
    
    def web_search(self, queries, num_results=3):
        url = 'https://www.googleapis.com/customsearch/v1'
        all_links = []
        if isinstance(queries, str):
            num_results *= 2
            queries = [queries]
        for query in queries:
            try:
                params = {
                    "key": os.getenv("GOOGLE_API_KEY"),
                    "cx": os.getenv("GOOGLE_CX_ID"),
                    "q": query,
                    "num": num_results
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    results = response.json()
                    all_links.extend([item["link"] for item in results.get('items', [])])
            except Exception as e:
                print(e)
        return list(set(all_links))
    
    def get_processed_texts(self, file):
        all_chunks = []
        all_sources = []
        all_pages = self.splitter.split_documents(self.remove_empty_space(file))
        for page in all_pages:
            all_chunks.append(page.page_content)
            all_sources.append(page.metadata["source"])
        return all_chunks, all_sources
import json
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import boto3
import feedparser
import lancedb
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.embeddings.bedrock import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.lancedb import LanceDB
from loguru import logger
from requests.exceptions import RequestException
from transformers import AutoTokenizer
from unstructured.cleaners.core import clean_non_ascii_chars, clean_postfix
from unstructured.partition.html import partition_html

logger.add(f"logs/{Path(__file__).stem}_" + "{time}.log", backtrace=True, diagnose=True)


# Function to get total number of records in table
def get_total_records():
    table_name = st.session_state.lancedb_table_name
    db = lancedb.connect(st.session_state.vectorstore_path)
    table = db.open_table(table_name)
    records = table.search().limit(10000).to_list()
    logger.info(f"Total records in {table_name} = {len(records)}")
    return len(records)


# Function to split text into chunks
def split_text_to_chunks(text, tokenizer, max_length):
    """
    Splits the given text into chunks where each chunk has a maximum length, using a tokenizer.

    Args:
    text (str): The text to be chunked.
    tokenizer_model (str): Model to use for the tokenizer.
    max_length (int): The maximum length of each chunk.

    Returns:
    List[str]: A list of text chunks, each chunk not exceeding the max_length.
    """

    # Tokenize the text. The tokenizer automatically handles splitting into words/tokens.
    tokens = tokenizer.tokenize(text)
    # print(f"Num Tokens: {len(num_tokens)}; max_length={max_length}")
    # Gather token_ids upto max_length into one chunk. Add the rest to the next chunk.
    if len(tokens) >= max_length:
        token_chunks = [
            tokens[i : i + max_length] for i in range(0, len(tokens), max_length)
        ]
        # convert token chunks back to text
        text_chunks = [
            tokenizer.convert_tokens_to_string(chunk) for chunk in token_chunks
        ]
    else:
        text_chunks = [text]
    return text_chunks


def split_docs_with_tokenizer(docs: List[Document], tokenizer) -> List[Document]:
    doc_chunks = []
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=500, chunk_overlap=0
    )
    doc_chunks = text_splitter.split_documents(docs)
    return doc_chunks


# Extract text from html docs
def extract_text_chunks_from_html(urls, metadatas, html_docs) -> List[Document]:
    """ "
    Function to reformat html_docs from html to plain text
    Input: urls, html_docs
    Output: List[Document]
    """
    extracted_docs = []
    total_docs = len(urls)
    progress_count = 0
    ext_progress_bar = st.progress(progress_count, text="Converting html to text")
    for idx, (url, metadata, doc) in enumerate(zip(urls, metadatas, html_docs)):
        elements = partition_html(
            text=doc.page_content,
            html_assemble_articles=True,
            skip_headers_and_footers=True,
            chunking_strategy="by_title",
        )
        extracted_text = "".join([e.text for e in elements])
        extracted_text = clean_postfix(
            extracted_text, pattern="\n\nComments\n\nView Comments"
        )
        doc.page_content = clean_non_ascii_chars(extracted_text)
        doc.metadata = metadata
        extracted_docs.append(doc)
        progress_count = int((idx / total_docs) * 100)
        ext_progress_bar.progress(
            progress_count, text=f"Extracting plain text from {url}"
        )
    ext_progress_bar.empty()
    return extracted_docs


# Function to chunk, encode and add documents to LanceDB
def add_documents_to_lancedb(links, metadatas, html_docs):
    extracted_docs = extract_text_chunks_from_html(links, metadatas, html_docs)

    tokenizer = AutoTokenizer.from_pretrained("Cohere/Cohere-embed-english-v3.0")
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=500, chunk_overlap=0
    )
    doc_chunks = text_splitter.split_documents(extracted_docs)
    # logger.info(st.session_state['embedding_model_name'])
    region = "us-west-2"
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    embeddings = BedrockEmbeddings(
        client=bedrock_client, model_id=st.session_state.embedding_model_name, region_name=region
    )

    lancedb_uri = st.session_state.vectorstore_path
    db = lancedb.connect(lancedb_uri)
    table_name = st.session_state.lancedb_table_name
    table = db.open_table(table_name)
    num_records = len(doc_chunks)
    logger.info(f"Adding {num_records} to LanceDB table: {table_name}")
    vectorstore = LanceDB(
        connection=table,
        embedding=embeddings,
        vector_key="vector",
        id_key="id",
        text_key="text",
    )

    with st.spinner(f"Adding {num_records} to table: {table_name}, please wait ..."):
        _ = vectorstore.from_documents(
            documents=doc_chunks, embedding=embeddings, connection=table
        )
    st.toast(f"Added {num_records} records into table: {table_name} on {lancedb_uri}")
    return num_records


# Function to get html text from url
def get_html_text(url):
    DATADIR = Path("./data")
    parsed_url = urlparse(url)
    folder_name = parsed_url.netloc.replace(".", "_")
    DATADIR = DATADIR.joinpath(folder_name)
    DATADIR.mkdir(exist_ok=True, parents=True)
    file_name = parsed_url.path.rstrip("/").split("/")[-1]
    if DATADIR.joinpath(file_name).exists():
        with open(DATADIR.joinpath(file_name), "r") as f:
            return f.read()
    try:
        response = requests.get(url)
        response.raise_for_status()
    except RequestException as e:
        st.error("Error during requests to {0} : {1}".format(url, e))
        return None
    html_content = BeautifulSoup(response.content, "html.parser")
    DATADIR.joinpath(file_name).write_text(str(html_content), encoding="utf-8")
    return str(html_content)


# Function to scrape data from RSS feed
def scrape_rss_feeds(feed_urls):
    links = []
    metadatas = []
    html_docs = []
    feed_progress_count = 0
    feed_progress_bar = st.progress(
        feed_progress_count, text="Extracting links from feeds..."
    )
    total_feed_urls = len(feed_urls)
    for fidx, blog_feed in enumerate(feed_urls):
        feed = feedparser.parse(blog_feed)
        feed_progress_count = int((fidx / total_feed_urls) * 100)
        feed_progress_bar.progress(
            feed_progress_count, text=f"Extracting links from {blog_feed}"
        )
        entry_progress_count = 0
        entry_progress_text = "Extracting links, metadata from feeds..."
        entry_progress_bar = st.progress(entry_progress_count, text=entry_progress_text)
        total_feed_entries = len(feed.entries)
        for idx, entry in enumerate(feed.entries):
            metadata = dict()
            link = entry.link
            entry_progress_text = f"Extracting content from {link}..."
            entry_progress_count = int((idx / total_feed_entries) * 100)
            entry_progress_bar.progress(entry_progress_count, text=entry_progress_text)
            html_content = get_html_text(link)
            metadata["published"] = entry.published
            metadata["title"] = entry.title
            metadata["source"] = link
            # check if entry has key names authors, summary. Add keys to metadata accordingly.
            authors = entry.authors if "authors" in entry else []
            if authors:
                authors = [a["name"] for a in authors]  # expand author dictionaries
                metadata["authors"] = authors
            summary_text = entry.summary if "summary" in entry else ""
            if len(summary_text) > 0:
                # Summaries have tags in em sometimes. Clean em before adding.
                metadata["summary"] = "".join(
                    [el.text for el in partition_html(text=summary_text)]
                )
            html_doc = Document(page_content=html_content, metadata=metadata)
            links.append(link)
            metadatas.append(metadata)
            html_docs.append(html_doc)
        entry_progress_bar.empty()
    feed_progress_bar.empty()
    return links, metadatas, html_docs


# Function to scrape data from single, multiple URLs
def scrape_urls(urls):
    links = []
    metadatas = []
    html_docs = []
    total_urls = len(urls)
    progress_count = 0
    progress_text = "Extracting content URLs..."
    urls_progress_bar = st.progress(progress_count, text=progress_text)
    for idx, link in enumerate(urls):
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        html_content = soup.prettify()
        title = soup.title.string if soup.title else "No Title Found"
        published_date = soup.find("meta", property="article:published_time")

        metadata = dict()
        metadata["published"] = (
            published_date.get("content") if published_date else "No Data"
        )
        metadata["title"] = title
        metadata["source"] = link
        metadata["summary"] = "No data"
        metadata["authors"] = ["No data"]
        links.append(link)
        metadatas.append(metadata)
        html_doc = Document(page_content=html_content, metadata=metadata)
        # print(html_doc)
        # html_loader = async AsyncHtmlLoader(link)
        # html_doc = html_loader.load()
        html_docs.append(html_doc)
        progress_count = int((idx / total_urls) * 100)
        progress_text = f"Extracting content from:  {link}"
        urls_progress_bar.progress(progress_count, text=progress_text)
    # set progress bar to empty
    urls_progress_bar.empty()
    return links, metadatas, html_docs


# Function to scrape data from input fields
def scrape_data(rss_feed_urls, multiple_urls):
    # check for values in single_url, multiple_urls, then merge both of them into a single list
    single_urls = []
    all_feed_urls = []
    if multiple_urls:
        single_urls.extend(multiple_urls.split(","))
    if rss_feed_urls:
        logger.info(f"Received RSS: {rss_feed_urls}")
        all_feed_urls.extend(rss_feed_urls.split(","))
    # remove duplicate entries from single_urls and all_feed_urls
    single_urls = list(set(single_urls))
    all_feed_urls = list(set(all_feed_urls))
    links = []
    metadatas = []
    html_docs = []
    if len(single_urls) > 0:
        st.write(f"Scraping data from {len(single_urls)} URLs")
        links, metadatas, html_docs = scrape_urls(single_urls)
    if len(all_feed_urls) > 0:
        st.write(f"Scraping data from {len(all_feed_urls)} RSS Feeds")
        all_feed_urls = [
            url.strip().replace("\n", "") for url in all_feed_urls
        ]  # remove leading and trailing whitespaces]
        logger.info(f"Received URLs: {all_feed_urls}")
        links, metadatas, html_docs = scrape_rss_feeds(all_feed_urls)
    return links, metadatas, html_docs


# streamlit page for data ingestion with input components for ingesting RSS feeds, single, multiple URLs
def app():
    st.set_page_config(
        page_title="Add custom data to VectorStore", page_icon="✚📈", layout="wide"
    )
    st.title("Add blog data ⚙️")
    st.caption("Enter URLs to scrape and ingest data to VectorStore.")

    config_path = Path(st.session_state.config_file_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        for k, v in config.items():
            st.session_state[k] = v
    logger.info(st.session_state)
    # Input text field to Add RSS feed URLs
    st.subheader("Add content from RSS Feeds")
    rss_feed_urls = st.text_area(
        "Enter RSS Feed URLs only",
        key="rss_feed_urls",
        help="Enter valid RSS Feed URLs: comma separated",
        placeholder="Enter valid RSS Feed URLs - comma separated",
    )
    st.markdown("---")

    # Input text area to Add multiple URLs
    st.subheader("Add content from non RSS feed URLs")
    multiple_urls = st.text_area(
        "Enter multiple URLs",
        key="multiple_urls",
        help="Enter single URL or multiple URLs: comma separated",
        placeholder="Enter single URL or multiple URLs: comma separated",
    )
    st.markdown("---")

    # Bold text to display text, current num of records in the database in 2 columns
    col1, col2 = st.columns(2)
    col1.subheader("Number of records :")
    total_records_placeholder = col2.empty()

    # get total records in the database
    if "total_records" not in st.session_state:
        total_records = get_total_records()
        st.session_state.total_records = total_records

    total_records_placeholder.subheader(st.session_state.total_records)
    embedding_model_name = st.session_state.embedding_model_name
    # Add button to submit data
    if st.button("Submit Data"):
        # logger.info(f"Session variables: {st.session_state}")
        st.session_state.embedding_model_name = embedding_model_name
        # call function to scrape data based on input URLs
        links, metadatas, html_docs = scrape_data(rss_feed_urls, multiple_urls)
        # # call function to add data to LanceDb
        added_records = add_documents_to_lancedb(links, metadatas, html_docs)
        total_records = get_total_records()
        total_records_placeholder.subheader(total_records)
        st.session_state.total_records = total_records
        st.success(f"{added_records} records added to LanceDB successfully.")


if __name__ == "__main__":
    app()

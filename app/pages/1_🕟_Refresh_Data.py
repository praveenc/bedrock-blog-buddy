import sys
from pathlib import Path

import lancedb
import pandas as pd
import streamlit as st
from bedrock_utils import get_langchain_bedrock_embeddings
from blog_utils import ScrapeAWSBlogs
from langchain.vectorstores.lancedb import LanceDB
from loguru import logger

module_path = ".."
sys.path.append(str(Path(module_path).absolute()))

logger.add(
    f"logs/{Path(__file__).stem}.log", rotation="1 week", backtrace=True, diagnose=True
)


# Function to get total number of records in table
def get_total_records():
    table_name = st.session_state.lancedb_table_name
    db = lancedb.connect(st.session_state.vectorstore_path)
    table = db.open_table(table_name)
    records = table.search().limit(10000).to_list()
    logger.info(f"Total records in {table_name} = {len(records)}")
    return len(records)


# Function to chunk, encode and add documents to LanceDB
def add_documents_to_lancedb(doc_chunks, model_id):
    embeddings = get_langchain_bedrock_embeddings(
        model_id=model_id, region=st.session_state.aws_region
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


# streamlit page for data ingestion with input components for ingesting RSS feeds, single, multiple URLs
def app():
    st.set_page_config(
        page_title="Refresh AWS RSS blog feed", page_icon="✚📈", layout="wide"
    )
    st.title("Refresh blog RSS feeds ⚙️")
    st.caption("Click Refresh to get latest blog posts from RSS feeds.")

    # write session state to config.json
    for k, v in st.session_state.items():
        if k == "llm_model_name":
            # logger.info(f"Adding to session k: {k}, v: {v}")
            st.session_state.llm_model_name = v
        if k == "embedding_model_name":
            # logger.info(f"Adding to session k: {k}, v: {v}")
            st.session_state.embedding_model_name = v
        if k == "aws_region":
            st.session_state.aws_region = v

    with st.sidebar:
        st.subheader("**Configuration**")
        st.markdown(f"**Embedding Model:**`{st.session_state.embedding_model_name}`")
        st.markdown(f"**LLM:** `{st.session_state.llm_model_name}`")
        st.markdown(f"**AWS region:** `{st.session_state.aws_region}`")

    logger.info(st.session_state)
    # Input text field to Add RSS feed URLs
    st.subheader("Refresh data from RSS feed")

    aws_blog_feeds = {
        "machinelearning": "https://aws.amazon.com/blogs/machine-learning/feed/",
        "security": "https://aws.amazon.com/blogs/security/feed/",
        "bigdata": "https://aws.amazon.com/blogs/big-data/feed/",
        "containers": "https://aws.amazon.com/blogs/containers/feed/",
        "serverless": "https://aws.amazon.com/blogs/compute/tag/serverless/feed/",
        "operations": "https://aws.amazon.com/blogs/mt/feed/",
        "opensource": "https://aws.amazon.com/blogs/opensource/feed/",
    }

    feed_df = pd.DataFrame(aws_blog_feeds.items(), columns=["Feed", "URL"], index=None)
    st.table(feed_df)
    st.markdown("---")

    # Bold text to display text, current num of records in the database in 2 columns
    col1, col2 = st.columns(2)
    col1.subheader("Records in DB :")
    total_records_placeholder = col2.empty()

    # get total records in the database
    if "total_records" not in st.session_state:
        total_records = get_total_records()
        st.session_state.total_records = total_records

    total_records_placeholder.subheader(st.session_state.total_records)
    st.markdown("---")

    # Add button to submit data
    if st.button("Refresh feed"):
        all_extracted_docs = []
        for k, v in aws_blog_feeds.items():
            # logger.info(f"Refreshing feed for {k} category...")
            # logger.info(f"Feed URL: {v}")
            with st.spinner(f"Refreshing feed for {k} category..."):
                scraper = ScrapeAWSBlogs(feed_url=v, blog_category=k)
                extracted_docs = scraper.get_processed_docs()
                all_extracted_docs.extend(extracted_docs)
        # call function to scrape data based on input URLs
        docs_to_add = len(all_extracted_docs)
        logger.info(f"Total docs to add: {docs_to_add}")
        # links, metadatas, html_docs = ScrapeAWSBlogs(feed_url=)
        # # call function to add data to LanceDb
        if docs_to_add >= 1:
            added_records = add_documents_to_lancedb(
                doc_chunks=all_extracted_docs,
                model_id=st.session_state.embedding_model_name,
            )
            total_records = get_total_records()
            total_records_placeholder.subheader(total_records)
            st.session_state.total_records = total_records
            st.success(f"{added_records} records added to LanceDB successfully.")
        else:
            st.info("All RSS feeds are up to date! ✅")


if __name__ == "__main__":
    app()

import sys
from pathlib import Path

import lancedb
import streamlit as st
from bedrock_utils import get_langchain_bedrock_embeddings
from blog_utils import ScrapeAWSBlogs
from langchain.vectorstores.lancedb import LanceDB
from loguru import logger

module_path = ".."
sys.path.append(str(Path(module_path).absolute()))

logger.add(f"logs/{Path(__file__).stem}_" + "{time}.log", backtrace=True, diagnose=True)


# Function to get total number of records in table
def get_total_records():
    table_name = st.session_state.lancedb_table_name
    db = lancedb.connect(st.session_state.vectorstore_path)
    table = db.open_table(table_name)
    records = table.search().limit(10000).to_list()
    logger.info(f"Total records in {table_name} = {len(records)}")
    return len(records)


# Function to chunk, encode and add documents to LanceDB
def add_documents_to_lancedb(doc_chunks):
    model_id = [mdl for mdl in st.session_state.bedrock_embeddings if "english" in mdl][
        0
    ]
    # if model_id:
    #     model_id = model_id[0]
    print(model_id)
    embeddings = get_langchain_bedrock_embeddings(model_id=model_id, region="us-west-2")
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
        page_title="Add custom data to VectorStore", page_icon="✚📈", layout="wide"
    )
    st.title("Add blog data ⚙️")
    st.caption("Enter URLs to scrape and ingest data to VectorStore.")

    # write session state to config.json
    for k, v in st.session_state.items():
        if k == "llm_model_name":
            logger.info(f"Adding to session k: {k}, v: {v}")
            st.session_state.llm_model_name = v
        if k == "embedding_model_name":
            logger.info(f"Adding to session k: {k}, v: {v}")
            st.session_state.embedding_model_name = v

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

    # # Input text area to Add multiple URLs
    # st.subheader("Add content from non RSS feed URLs")
    # multiple_urls = st.text_area(
    #     "Enter multiple URLs",
    #     key="multiple_urls",
    #     help="Enter single URL or multiple URLs: comma separated",
    #     placeholder="Enter single URL or multiple URLs: comma separated",
    # )
    # st.markdown("---")

    # Bold text to display text, current num of records in the database in 2 columns
    col1, col2 = st.columns(2)
    col1.subheader("Number of records :")
    total_records_placeholder = col2.empty()

    # get total records in the database
    if "total_records" not in st.session_state:
        total_records = get_total_records()
        st.session_state.total_records = total_records

    total_records_placeholder.subheader(st.session_state.total_records)
    # embedding_model_name = st.session_state.embedding_model_name
    print(st.session_state.duckdb_path)
    # Add button to submit data
    if st.button("Submit Data"):
        # logger.info(f"Session variables: {st.session_state}")
        # print(st.session_state.embedding_model_name)
        rss_feed_urls = rss_feed_urls.split(",")
        if isinstance(rss_feed_urls, str):
            rss_feed_urls = [rss_feed_urls]
        all_extracted_docs = []
        for feed_url in rss_feed_urls:
            scraper = ScrapeAWSBlogs(
                feed_url=feed_url, duck_db_path=st.session_state.duckdb_path
            )
            extracted_docs = scraper.get_processed_docs()
            all_extracted_docs.extend(extracted_docs)
        # call function to scrape data based on input URLs
        # print(len(all_extracted_docs))
        # links, metadatas, html_docs = ScrapeAWSBlogs(feed_url=)
        # # call function to add data to LanceDb
        added_records = add_documents_to_lancedb(doc_chunks=all_extracted_docs)
        total_records = get_total_records()
        total_records_placeholder.subheader(total_records)
        st.session_state.total_records = total_records
        st.success(f"{added_records} records added to LanceDB successfully.")


if __name__ == "__main__":
    app()

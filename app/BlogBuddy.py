import sys
from pathlib import Path
from typing import List

import lancedb
import streamlit as st
from bedrock_utils import get_model_ids
from blog_utils import BlogsDuckDB
from lancedb.pydantic import LanceModel, Vector
from loguru import logger
from transformers import AutoTokenizer

module_path = "."
sys.path.append(str(Path(module_path).absolute()))

logger.add(f"logs/{Path(__file__).stem}.log", rotation="1 week", backtrace=True, diagnose=True)


def get_anthropic_llms(providers: List[str] = ["Anthropic"]):
    models = []
    for provider in providers:
        model_ids = get_model_ids(provider=provider, output_modality="TEXT")
        models.extend(model_ids)
    filtered_models = [mdl for mdl in models if "k" not in mdl]
    # st.session_state.bedrock_llms = models
    return filtered_models


def get_cohere_embedding_models(providers: List[str] = ["Cohere"]):
    models = []
    for provider in providers:
        model_ids = get_model_ids(provider=provider, output_modality="EMBEDDING")
        models.extend(model_ids)
    st.session_state.bedrock_embeddings = models
    return models


# Function to download HuggingFace embedding model and initialize tokenizer and model
def get_embedding_dimensions(embed_model_id="cohere.embed-english-v3"):
    dimensions = 512
    if embed_model_id == "cohere.embed-english-v3":
        dimensions = 1024
        st.session_state.dimensions = dimensions
    return dimensions


# Function to check if vectorstore_path exists, if not, create it
def check_vectorstore_path(vectorstore_path, dimensions, embed_model_id):
    if not vectorstore_path.exists():
        logger.info(f"Creating vectorstore_path: {vectorstore_path}")
        vectorstore_path.mkdir(parents=True, exist_ok=True)

    if "vectorstore_path" not in st.session_state:
        st.session_state.vectorstore_path = str(vectorstore_path)

    # Connect to LanceDB
    db = lancedb.connect(vectorstore_path)
    logger.info(f"Connected to LanceDB: {vectorstore_path}")
    # Create default table for mlblogs_jinav2base if it doesn't exist
    suffix = embed_model_id.replace(".", "_").replace("-", "_")
    table_name = f"mlblogs_{suffix}"
    if table_name not in db.table_names():
        # Define table schema using Pydantic
        class MLBlogs(LanceModel):
            id: str
            source: str
            title: str
            published: str
            summary: str = None
            text: str
            authors: List[str] = None
            vector: Vector(dim=dimensions)  # Vector dimensions

        table = db.create_table(table_name, schema=MLBlogs.to_arrow_schema())
        st.write(f"Created Table: {table_name}")
    else:
        st.markdown(f"**Table:** `{table_name}` already exists in LanceDB.")
        table = db.open_table(table_name)
    st.session_state.lancedb_table_name = table_name
    st.markdown(f"**Table `{table_name}` Schema:**")
    st.code(table.schema)


# Streamlit app for setting ChatApp configuration
def app():
    st.title("👋 Welcome to BlogBuddy 🤖")
    st.caption("To start, Select a LLM and an Embedding Model and click `Save`.")

    # Configuration options for setting the embedding model
    st.subheader("Embedding Model")
    # Input field for the embedding model name from Bedrock
    embed_model_names = get_cohere_embedding_models()
    embedding_model_name = st.selectbox(
        "Select embedding model",
        options=embed_model_names,
        key="embedding_model_name",
        help="Select an embedding model to be used for chat application.",
    )

    st.markdown("---")

    # Configuration options for selecting LLM to be used for chat application
    st.subheader("LLM")

    model_names = get_anthropic_llms()
    llm_model_name = st.selectbox(
        "Select Text generation model",
        options=model_names,
        key="llm_model_name",
        placeholder="Select a text generation model.",
        help="Select text generation model to be used for chat application.",
        # on_change=store_model_name,
    )
    st.markdown("---")

    st.subheader("AWS region")
    aws_region = st.selectbox(
        "Select AWS region",
        options=["us-east-1", "us-west-2"],
        key="aws_region",
        help="Select AWS region to be used for chat application.",
    )

    vectorstore_path = Path("./lancedb").absolute()
    duckdb_path = Path("./duckdb").absolute()
    duckdb_path.mkdir(exist_ok=True, parents=True)

    # Save button to store all the selected values to a file named config.json
    if st.button("Save"):
        # logger.info(st.session_state)
        # download embedding model if doesn't exist
        _ = get_embedding_dimensions(embedding_model_name)
        # logger.info(f"Dimensions: {dimensions}")
        # check if vectorstore_path exists, if not, create it
        vectorstore_path = Path(vectorstore_path).absolute()

        check_vectorstore_path(
            vectorstore_path,
            st.session_state.dimensions,
            st.session_state.embedding_model_name,
        )

        # create Duckdb table if not exist
        logger.info(f"Creating DuckDB table: {duckdb_path}")
        _ = BlogsDuckDB(duckdb_path)
        if "duckdb_path" not in st.session_state:
            st.session_state.duckdb_path = str(duckdb_path)

        # save the values to a file named config.json
        # print(vectorstore_path)
        if "embeddings_max_length" not in st.session_state:
            tokenizer = AutoTokenizer.from_pretrained(
                "Cohere/Cohere-embed-english-v3.0"
            )
            st.session_state.embeddings_max_length = tokenizer.model_max_length

        if "llm_model_name" not in st.session_state:
            st.session_state.llm_model_name = llm_model_name

        if "aws_region" not in st.session_state:
            st.session_state.aws_region = aws_region

        logger.info(st.session_state)

        with st.sidebar:
            st.subheader("**Configuration**")
            st.markdown(f"**Embedding Model:**`{embedding_model_name}`")
            st.markdown(f"**LLM:** `{llm_model_name}`")
            st.markdown(f"**AWS region:** `{aws_region}`")


if __name__ == "__main__":
    app()

import json
import warnings
from pathlib import Path

import lancedb
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.ollama import Ollama
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.lancedb import LanceDB
from loguru import logger
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

logger.add(f"logs/{Path(__file__).stem}_" + "{time}.log", backtrace=True, diagnose=True)

PROMPT_PREFIX = "[INST] "
PROMPT_SUFFIX = " [/INST]"
LLAMA2_PROMPT = """[INST] <<SYS>> {sys_prompt} <</SYS> {prompt} [/INST]"""


# Function to get total number of records in table
def get_total_records(db_path, table_name):
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    records = table.search().limit(10000).to_list()
    return len(records)


# Function to get embeddings function
def get_embeddings_function(model_id):
    # HF_MODEL_ID = "jinaai/jina-embeddings-v2-base-en"
    _ = AutoTokenizer.from_pretrained(model_id)
    _ = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    model_kwargs = {"device": "mps"}
    encode_kwargs = {"normalize_embeddings": True}
    jina_embeddings = HuggingFaceEmbeddings(
        model_name=model_id, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    print("Inside get_embeddings_function")
    return jina_embeddings


# Function to get retriever connection to LanceDB vectorstore
def get_retriever(db_path, table_name, embed_model_id, topk):
    print("Inside get_retriever")
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    # connect to vector store
    _ = AutoTokenizer.from_pretrained(embed_model_id)
    _ = AutoModel.from_pretrained(embed_model_id, trust_remote_code=True)

    model_kwargs = {"device": "mps"}
    encode_kwargs = {"normalize_embeddings": True}
    jina_embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    vectorstore = LanceDB(connection=table, embedding=jina_embeddings)
    # set vectorstore as retriever
    retriever_kwargs = {
        "search_type": "similarity",
        "search_kwargs": {"k": topk},
    }
    retriever = vectorstore.as_retriever(**retriever_kwargs)
    return retriever


# Function to return ollama llm
def get_llm(model_name):
    llm = Ollama(model=model_name)
    return llm


# Streamlit Chatbot app
def app():
    st.set_page_config(page_title="Private GPT Chatbot", page_icon="💬")
    st.title("An Chatbot using Ollama")
    st.caption("🚀 A private ollama chatbot")
    # Read config file path from session state
    config_path = Path(st.session_state.config_file_path)
    print(config_path)
    # config_path = Path("../config.json")
    if config_path.exists():
        with open("config.json", "r") as f:
            config = json.load(f)
        # Load the values from the config.json file and assign as session_state variable
        st.session_state.embedding_model_name = config["embedding_model_name"]
        st.session_state.llm_model_name = config["llm_model_name"]
        st.session_state.vectorstore_name = config["vectorstore_name"]
        st.session_state.vectorstore_path = config["vectorstore_path"]
        st.session_state.dimensions = config["dimensions"]
        st.session_state.ollama_models = config["ollama_models"]
        st.session_state.lancedb_table_name = config["lancedb_table_name"]
        st.session_state.embedding_max_length = config["embedding_max_length"]

    with st.sidebar:
        st.markdown(f"**Model:** {st.session_state.llm_model_name}")
        db_path = st.session_state.vectorstore_path
        table_name = st.session_state.lancedb_table_name
        st.markdown(f"**Table:** {table_name}")
        total_records = get_total_records(db_path, table_name)
        st.markdown(f"**Total Records:** _{total_records}_")
        st.markdown("---")

    # Initialize Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    # React to user input
    if prompt := st.chat_input():
        # add users promt to session messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        st.chat_message("user").write(prompt)

        # Prepare the prompt and invoke the chain to get output from llm
        db_path = st.session_state.vectorstore_path
        model_name = st.session_state.llm_model_name
        embed_model_id = st.session_state.embedding_model_name
        table_name = st.session_state.lancedb_table_name
        print(model_name)
        print(embed_model_id)
        print(table_name)
        llm = get_llm(model_name)
        # retriever = get_retriever(
        #     db_path=db_path,
        #     table_name=table_name,
        #     embed_model_id=embed_model_id,
        #     topk=3,
        # )
        # json_output prompt
        # llama2_prompt = PromptTemplate.from_file(
        #     template_file=Path(
        #         f"./prompts/{model_name}/json_prompt_{model_name}.txt"
        #     ).resolve(),
        #     input_variables=["text"],
        # )
        # document summarizer prompt
        llama2_prompt = PromptTemplate.from_file(
            template_file=Path(
                f"./prompts/{model_name}/doc_summarizer_{model_name}.txt"
            ).resolve(),
            input_variables=["document"],
        )

        # RAG prompt
        # llama2_prompt = PromptTemplate.from_file(
        #     template_file=Path(
        #         f"./prompts/{model_name}/rag_prompt_{model_name}.txt"
        #     ).resolve(),
        #     input_variables=["context", "question"],
        # )
        chain = (
            {"document": RunnablePassthrough()}
            | llama2_prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner(f'Generating text {model_name} Please wait ...'):
            output = chain.invoke(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(output)
            ai_log = {
                "role": "assistant",
                "content": f"""{output}""",
            }
            st.session_state.messages.append(ai_log)


if __name__ == "__main__":
    app()

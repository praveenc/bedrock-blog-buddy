import sys
import warnings
from pathlib import Path

import lancedb
import streamlit as st
from bedrock_utils import (
    get_langchain_bedrock_embeddings,
    get_langchain_bedrock_llm,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.lancedb import LanceDB
from loguru import logger

module_path = ".."
sys.path.append(str(Path(module_path).absolute()))

warnings.filterwarnings("ignore")

logger.add(
    f"logs/{Path(__file__).stem}.log", rotation="1 week", backtrace=True, diagnose=True
)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"


# Function to get total number of records in table
def get_lancedb_records(db_path, table_name):
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    records = table.search().limit(10000).to_list()
    # logger.info(f"Total records in {table_name} = {len(records)}")
    return len(records)


# Function to get retriever connection to LanceDB vectorstore
def get_retriever(db_path, table_name, topk=3):
    print("Inside get_retriever")
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    model_id = st.session_state.embedding_model_name
    embeddings = get_langchain_bedrock_embeddings(model_id=model_id, region="us-west-2")
    vectorstore = LanceDB(connection=table, embedding=embeddings)
    # set vectorstore as retriever
    retriever_kwargs = {
        "search_type": "similarity",
        "search_kwargs": {"k": topk},
    }
    retriever = vectorstore.as_retriever(**retriever_kwargs)
    return retriever


# function to format the retrieved docs into xml tags for claude
def format_context_docs(docs):
    context_string = ""
    for idx, _d in enumerate(docs):
        metadata = _d.metadata
        otag = f"<document index={idx+1}>"
        ctag = "</document>"
        src_text = f"<source>{metadata['source']}</source>"
        c_text = f"{otag}<document_content>{_d.page_content}</document_content>{src_text}{ctag}\n"
        context_string += c_text
    # print(context_string)
    return context_string


# Streamlit Chatbot app
def app():
    # st.set_page_config(page_title="Private GPT Chatbot", page_icon="💬")
    st.title("Chat with AWS Blog Posts")
    st.caption("🚀 Chat with top 20 blog posts from AWS Blogs RSS feeds.")

    # add custom session_state items
    for k, v in st.session_state.items():
        if k == "llm_model_name":
            st.session_state.llm_model_name = v
        if k == "embedding_model_name":
            st.session_state.embedding_model_name = v
        if k == "aws_region":
            st.session_state.aws_region = v

    total_records = get_lancedb_records(
        db_path=st.session_state.vectorstore_path,
        table_name=st.session_state.lancedb_table_name,
    )

    with st.sidebar:
        st.subheader("**Configuration**")
        st.markdown(f"**Embedding Model:**`{st.session_state.embedding_model_name}`")
        st.markdown(f"**LLM:** `{st.session_state.llm_model_name}`")
        st.markdown(f"**AWS region:** `{st.session_state.aws_region}`")
        st.markdown("---")
        st.subheader("**Records in DB**")
        st.markdown(f"**{total_records}**")

    # Initialize Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.chat_message("assistant").write("Hi, welcome to blog buddy")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    db_path = st.session_state.vectorstore_path
    model_name = st.session_state.llm_model_name
    table_name = st.session_state.lancedb_table_name

    # React to user input
    if prompt := st.chat_input():
        # add users promt to session messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        st.chat_message("user").write(prompt)

        llm = get_langchain_bedrock_llm(model_id=model_name, region="us-west-2")
        retriever = get_retriever(
            db_path=db_path,
            table_name=table_name,
            topk=3,
        )
        if model_name.split("-")[-1] == "v2:1":
            prompt_file = Path("prompts/rag_prompt_v2_1.txt").absolute()
        else:
            prompt_file = Path("prompts/rag_prompt_v2.txt").absolute()

        print(prompt_file)
        rag_prompt = PromptTemplate.from_file(
            prompt_file, input_variables=["context", "question"]
        )
        rag_chain = (
            {
                "question": RunnablePassthrough(),
                "context": retriever | format_context_docs,
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner(f"Generating using {model_name} ..."):
            output = rag_chain.invoke(prompt)
            output = output.replace("</answer>", "")
            # logger.info(f"LLM Output: {output}")

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

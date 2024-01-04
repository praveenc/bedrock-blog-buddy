from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import duckdb
import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from loguru import logger
from requests import RequestException
from spacy.lang.en import English
from tqdm import tqdm
from transformers import AutoTokenizer
from unstructured.cleaners.core import clean_non_ascii_chars, clean_postfix
from unstructured.partition.html import partition_html

logger.add(f"logs/{Path(__file__).stem}_" + "{time}.log", backtrace=True, diagnose=True)


class BlogsDuckDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)
        self.table_name = "blogposts"
        _ = self.create_blogposts_table()

    def get_connection(self):
        return self.conn

    def close_connection(self):
        return self.conn.close()

    def create_blogposts_table(self):
        self.conn.sql("CREATE SEQUENCE IF NOT EXISTS blogid_seq START 1;")
        return self.conn.sql(
            f"CREATE TABLE IF NOT EXISTS {self.table_name}(id INTEGER PRIMARY KEY DEFAULT NEXTVAL('blogid_seq'), blog_domain VARCHAR, blogpost_url VARCHAR, date_published TIMESTAMP)"
        )

    def show_tables(self):
        tables_df = self.conn.execute("SHOW ALL TABLES;").fetchdf()
        if len(tables_df) == 0:
            logger.info("No tables")
            return
        return tables_df

    def insert_record(self, df: pd.DataFrame):
        result = self.conn.execute(f"INSERT INTO {self.table_name} SELECT * FROM df")
        if result is not None:
            return result.fetchdf()
        return pd.DataFrame()

    def delete_record_with_id(
        self,
        record_id: int,
    ) -> pd.DataFrame:
        result = self.conn.execute(
            f"DELETE FROM {self.table_name} WHERE id = {record_id}"
        )
        if result is not None:
            return result.fetchdf()
        return pd.DataFrame()

    def get_all_records(self) -> pd.DataFrame:
        result = self.conn.execute(f"SELECT * FROM {self.table_name}")
        if result is not None:
            return result.fetch_df()
        return pd.DataFrame()

    def delete_all_records(self):
        results = self.conn.sql(f"DELETE FROM {self.table_name}")
        return results

    def query(self, query: str):
        df = self.conn.execute(query).fetch_df()
        return df

    def dump(self, sql: str):
        df = self.conn.execute(sql).fetch_df()
        return df

    def close(self):
        self.conn.close()


class ScrapeAWSBlogs:
    def __init__(
        self,
        feed_url: str,
        duck_db_path: str,
        target_dir: Path = Path("./data"),
        table_name: str = "blogposts",
    ) -> None:
        self.rss_feed_url = feed_url
        if Path(duck_db_path).exists():
            self.duckdb_conn = BlogsDuckDB(str(duck_db_path)).get_connection()
        else:
            logger.error(f"DuckDB database not found at {duck_db_path}")
            raise FileNotFoundError
            sys.exit(-1)
        self.table_name = table_name
        self.target_dir = target_dir
        self.target_dir.mkdir(exist_ok=True, parents=True)
        self.is_download = True
        nlp = English()
        nlp.add_pipe("sentencizer")
        self.nlp = nlp
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Cohere/Cohere-embed-english-v3.0"
        )
        self.max_length = self.tokenizer.model_max_length

    def is_url_in_db(self, url: str) -> bool:
        results = self.duckdb_conn.sql(
            f"SELECT * FROM {self.table_name} WHERE blogpost_url='{url.strip()}'"
        ).fetchall()
        if len(results) >= 1:
            return True
        return False

    def get_html_text(self, url: str) -> str:
        """
        Function to download html page to disk and return the html content.
        If the page is already downloaded then skips scraping again.
        """
        parsed_url = urlparse(url)
        folder_name = parsed_url.netloc.replace(".", "_")
        DATADIR = self.target_dir.joinpath(folder_name)
        file_name = parsed_url.path.rstrip("/").split("/")[-1]
        if DATADIR.joinpath(file_name).exists():
            # logger.info(f"Reading {file_name} from disk")
            with open(DATADIR.joinpath(file_name), "r") as f:
                return f.read()
        try:
            response = requests.get(url)
            response.raise_for_status()
        except RequestException as e:
            logger.error(f"Error during requests to {url} : {e}")
            return None
        html_content = BeautifulSoup(response.content, "html.parser")
        DATADIR.joinpath(file_name).write_text(str(html_content), encoding="utf-8")
        return str(html_content)

    def scrape_rss_feed(self):
        links = []
        metadatas = []
        html_docs = []
        rss_feed = feedparser.parse(self.rss_feed_url)
        for entry in tqdm(
            rss_feed.entries,
            desc="Extracting links, metadatas from feed",
            total=len(rss_feed.entries),
        ):
            metadata = dict()
            link = entry.link
            html_content = self.get_html_text(link)
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
            if not self.is_url_in_db(link):
                self.log_scrape_details(link, str(entry.published))
            # else:
            #     logger.info(f"Skipping DB entry: {link}")
        return links, metadatas, html_docs

    def get_processed_docs(self) -> List[Document]:
        """
        Function to reformat html_docs from html to plain text
        Input: urls, html_docs
        Output: List[Document]
        """
        urls, metadatas, html_docs = self.scrape_rss_feed()
        extracted_docs = []
        chunked_docs = []
        for url, metadata, doc in tqdm(
            zip(urls, metadatas, html_docs),
            desc="Extracting text from html",
            total=len(html_docs),
        ):
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

        for doc in extracted_docs:
            text_chunks = self.chunk_text(doc.page_content)
            doc_chunks = [
                Document(page_content=txt, metadata=doc.metadata) for txt in text_chunks
            ]
            chunked_docs.extend(doc_chunks)
        return chunked_docs

    def get_num_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text):
        doc = self.nlp(text)
        chunks = []
        current_chunk = ""
        for sentence in doc.sents:
            # Check the token length if this sentence is added
            if self.get_num_tokens(current_chunk + sentence.text) < self.max_length:
                current_chunk += sentence.text + " "
            else:
                # If adding the sentence exceeds the max_length, start a new chunk
                chunks.append(current_chunk)
                current_chunk = sentence.text + " "
        chunks.append(current_chunk)  # Add the last chunk
        return chunks

    def log_scrape_details(self, link: str, date_published: str):
        blog_domain = urlparse(link).netloc
        blogpost_url = link
        datetime_obj = datetime.strptime(date_published, "%a, %d %b %Y %H:%M:%S %z")
        final_dt = datetime.strftime(datetime_obj, "%Y-%m-%d %H:%M:%S")
        SQL = f"INSERT OR IGNORE INTO {self.table_name} VALUES (nextval('blogid_seq'),'{blog_domain}', '{blogpost_url}', '{final_dt}')"
        result_df = self.duckdb_conn.execute(SQL).fetchdf()
        return result_df

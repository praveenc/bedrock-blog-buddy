import sys
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
    def __init__(self, db_dir: Path):
        self.db_dir = db_dir
        self.db_name = "blogposts"
        self.table_name = "blogposts"
        if not db_dir.exists():
            db_dir.mkdir(exist_ok=True, parents=True)
        self.db_path = db_dir.joinpath(self.db_name)
        self.conn = duckdb.connect(str(self.db_path))
        self.create_blogposts_table()

    def get_table_name(self):
        return self.table_name

    def get_connection(self):
        return self.conn

    def close_connection(self):
        return self.conn.close()

    def create_blogposts_table(self):
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS blogid_seq START 1;")
        SQL = f"""CREATE TABLE IF NOT EXISTS
        {self.table_name}(
            id INTEGER PRIMARY KEY DEFAULT NEXTVAL('blogid_seq'),
            blog_domain VARCHAR,
            blog_category VARCHAR,
            blogpost_url VARCHAR,
            blog_summary VARCHAR,
            published_date TIMESTAMP
        )"""
        return self.conn.execute(SQL)

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

    def get_latest_posts(self, category, days=20) -> pd.DataFrame:
        SQL = f"""SELECT blog_category, blogpost_url, blog_summary, published_date FROM blogposts
        WHERE blog_category = '{category}'
        AND published_date > CURRENT_TIMESTAMP - INTERVAL '{days} days'"""
        result = self.conn.execute(SQL)
        if result is not None:
            return result.fetchdf()
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
        blog_category: str,
        duckdb_dir: Path = Path("duckdb"),
        target_dir: Path = Path("./data"),
    ) -> None:
        self.rss_feed_url = feed_url
        if duckdb_dir.exists():
            duckdb = BlogsDuckDB(duckdb_dir)
            self.duckdb_conn = duckdb.get_connection()
            self.table_name = duckdb.get_table_name()
        else:
            logger.error(f"DuckDB database not found at {duckdb_dir}")
            raise FileNotFoundError
            sys.exit(-1)

        self.target_dir = target_dir
        self.blog_category = blog_category
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
        extracts Blog post conclusion and removes text about authors.
        If the page is already downloaded then skips scraping again.
        Returns html_content and post summary if available
        """

        def clean_and_extract_summary(content):
            soup = BeautifulSoup(content, "html.parser")
            conclusion = soup.find("h2", text="Conclusion")
            conclusion_paragraphs = []
            if conclusion:
                for sibling in conclusion.find_next_siblings():
                    if sibling.name == "p":
                        conclusion_paragraphs.append(sibling.text)
                    if sibling.name == "h2" or sibling.name == "h3":
                        break

            # Find the tag and remove all subsequent siblings
            about_authors = soup.find("h3", text="About the Authors")
            if about_authors:
                for sibling in about_authors.find_next_siblings():
                    sibling.decompose()
                about_authors.decompose()
            cleaned_content = str(soup)
            conclusion_text = " ".join(conclusion_paragraphs)
            return cleaned_content, conclusion_text

        parsed_url = urlparse(url)
        folder_name = parsed_url.netloc.replace(".", "_")
        DATADIR = self.target_dir.joinpath(folder_name)
        DATADIR.mkdir(exist_ok=True, parents=True)
        file_name = parsed_url.path.rstrip("/").split("/")[-1]
        if DATADIR.joinpath(file_name).exists():
            with open(DATADIR.joinpath(file_name), "r") as f:
                file_content = f.read()
                html_content, conclusion = clean_and_extract_summary(file_content)
        else:
            try:
                response = requests.get(url)
                response.raise_for_status()
                html_content, conclusion = clean_and_extract_summary(response.content)
                DATADIR.joinpath(file_name).write_text(html_content, encoding="utf-8")
            except RequestException as e:
                logger.error(f"Error during requests to {url} : {e}")
                return "None", "None"
        return html_content, conclusion

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
            if self.is_url_in_db(link):
                # logger.info(f"Already processed, skipping {link}")
                continue
            html_content, summary = self.get_html_text(link)
            metadata["published"] = entry.published
            metadata["title"] = entry.title
            metadata["source"] = link
            metadata["summary"] = summary
            # check if entry has key names authors, summary. Add keys to metadata accordingly.
            authors = [a["name"] for a in entry.authors] if "authors" in entry else []
            metadata["authors"] = authors
            html_doc = Document(page_content=html_content, metadata=metadata)
            links.append(link)
            metadatas.append(metadata)
            html_docs.append(html_doc)
            self.log_scrape_details(link, str(entry.published), summary)
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

    def log_scrape_details(self, link: str, date_published: str, summary: str):
        blog_domain = urlparse(link).netloc
        datetime_obj = datetime.strptime(date_published, "%a, %d %b %Y %H:%M:%S %z")
        final_dt = datetime.strftime(datetime_obj, "%Y-%m-%d %H:%M:%S")
        SQL = f"""INSERT OR IGNORE INTO {self.table_name} VALUES (nextval('blogid_seq'), '{blog_domain}', '{self.blog_category}', '{link}', '{summary}', '{final_dt}')"""
        result_df = self.duckdb_conn.execute(SQL).fetchdf()
        return result_df

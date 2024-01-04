# A ChatBot for AWS Blogs 🤖 📋

A chatbot to talk to AWS Blogs.

## Usage

### Installation

```shell
git clone https://github.com/praveenc/bedrock-blog-buddy
```

```shell
cd bedrock-blog-buddy

pip install -r requirements.txt
```

### Launch streamlit app

```shell
cd app

# Ensure you're logged into aws using CLI.

streamlit run BlogBuddy.py
```

## VectorDB and Storage

We use [LanceDB](https://lancedb.com/) as our vector store. Vectors and metadata are stored locally.

To avoid scraping RSS feeds multiple times, we cache scraped html data to disk and log the scraping activity to [DuckDB](https://duckdb.org/docs/guides/python/install) locally.

## AWS Blogs RSS Feeds

Blog posts are indexed from the below AWS RSS feeds.

- [AWS Machine Learning blogs](https://aws.amazon.com/blogs/machine-learning/feed/)
- [AWS Security blogs](https://aws.amazon.com/blogs/security/feed)
- [AWS Analytics/Big-Data blogs](https://aws.amazon.com/blogs/big-data/feed/)
- [AWS Containers blogs](https://aws.amazon.com/blogs/containers/feed/)
- [AWS Serverless blogs](https://aws.amazon.com/blogs/compute/tag/serverless/feed/)
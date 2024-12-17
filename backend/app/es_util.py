from __future__ import annotations

import uuid
from typing import Any, Iterable, List
import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# 设置基本的配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElasticSearchBM25Retriever(BaseRetriever):
    """`Elasticsearch` retriever that uses `BM25`.
    """

    client: Any
    """Elasticsearch client."""
    index_name: str
    """Name of the index to use in Elasticsearch."""

    @classmethod
    def create(
        cls, elasticsearch_url: str, index_name: str, k1: float = 2.0, b: float = 0.75
    ) -> ElasticSearchBM25Retriever:
        """
        Create a ElasticSearchBM25Retriever from a list of texts.

        Args:
            elasticsearch_url: URL of the Elasticsearch instance to connect to.
            index_name: Name of the index to use in Elasticsearch.
            k1: BM25 parameter k1.
            b: BM25 parameter b.

        Returns:

        """
        from elasticsearch import Elasticsearch

        # Create an Elasticsearch client instance
        es = Elasticsearch(elasticsearch_url)

        # Define the index settings and mappings
        settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": k1,
                    "b": b,
                }
            },
        }
        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",  # Use the custom BM25 similarity
                }
            }
        }

        # Create the index with the specified settings and mappings
        es.indices.create(index=index_name, mappings=mappings, settings=settings)
        return cls(client=es, index_name=index_name)


    def add_texts(
        self,
        texts: Iterable[str],
        refresh_indices: bool = True,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the retriever.

        Args:
            texts: Iterable of strings to add to the retriever.
            refresh_indices: bool to refresh ElasticSearch indices

        Returns:
            List of ids from adding the texts into the retriever.
        """
        try:
            from elasticsearch.helpers import bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )
        requests = []
        ids = []
        for i, text in enumerate(texts):
            _id = str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": text,
                "_id": _id,
            }
            ids.append(_id)
            requests.append(request)
        bulk(self.client, requests)

        if refresh_indices:
            self.client.indices.refresh(index=self.index_name)
        return ids


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        keywords = query.split(",")
        # query_dict = {"query": {"match": {"content": query}}}
        should_conditions = [{"match_phrase": {"content": keyword}} for keyword in keywords]
        query_dict = {
            "query": {
                "bool": {
                    "should": should_conditions,
                    "minimum_should_match": 1
                }
            }
        }
        logger.info(f"ES Query: {query_dict}")
        res = self.client.search(index=self.index_name, body=query_dict)
        docs = []
        for r in res["hits"]["hits"]:
            logger.info(f"ES BM25 Score: {r['_score']}")
            # 检查文档得分是否满足阈值要求
            if r["_score"] >= 0.5:
                # 如果得分高于或等于阈值，则添加到文档列表
                docs.append(Document(page_content=r["_source"]["content"]))
        return docs

    
    
'''
from elasticsearch import Elasticsearch
from langchain_community.retrievers import ElasticSearchBM25Retriever

elasticsearch_url = "http://es01:9200"
es_client = Elasticsearch(elasticsearch_url)
# es_client = Elasticsearch(
#     [elasticsearch_url],
#     request_timeout=60,  # 连接超时
#     max_retries=3,
#     retry_on_timeout=True,
# )
retriever = ElasticSearchBM25Retriever(client=es_client, index_name="knowledge_base")
retriever.invoke('慶弔見舞金規程')
retriever.invoke(['規程'])
retriever.add_texts(["Hello, world!", "Goodbye, world!"])



def extract_key_nouns(text):
    # 加载 Ginza 日语模型
    nlp = spacy.load('ja_ginza')
    # 处理文本
    doc = nlp(text)
    # 提取名词
    nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
    return nouns
'''
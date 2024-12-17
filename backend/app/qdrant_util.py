from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct 
import os 
import uuid 
from typing import List
import logging

from .llm_util import generate_embeddings, generate_openai, generate_ollama, client, generate_embeddings_ollama

# 设置基本的配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def qdrant_get_points(client, chunks: List[str], is_openai=False):
    points = []
    for idx, chunk in enumerate(chunks):
        if is_openai:
            embeddings = generate_embeddings(client, chunk)
        else:
            embeddings = generate_embeddings_ollama(client, chunk)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings,
            payload={"text": chunk}
        )
        points.append(point)
    return points 

async def qdrant_insert_points(connection, points: List[PointStruct], collection_name="knowledge_base", is_openai=False):
    if is_openai:
        vector_dim = 1536
    else:
        vector_dim = 1024
    if not await connection.collection_exists(collection_name):
        # connection.delete_collection(collection_name)
        await connection.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE),
            hnsw_config={
                'm': 40,  # Max number of connections for each element (higher for larger datasets)
                'ef_construct': 800,  # Higher ef_construct improves index quality, but slows down indexing
                'full_scan_threshold': 10000  # Use full-scan for small collections
            }
        )
        
        operation_info = await connection.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )

async def qdrant_search_context(client, connection, query: str, collection_name="knowledge_base", is_openai=False):
    if is_openai:
        embeddings = generate_embeddings(client, query)
    else:
        embeddings = generate_embeddings_ollama(client, query)
    search_results = await connection.search(
        collection_name=collection_name,
        query_vector=embeddings,
        score_threshold=0.3,
        limit=3,
        search_params=models.SearchParams(hnsw_ef=200, exact=False),
    )
    context = []
    for result in search_results:
        logger.info(f"Searched text: {result}")
        context.append(result.payload["text"])
    return context 


if __name__ == "__main__":
    connection = QdrantClient("http://qdrant:6333")
    pdf_path = "../assets/compact-guide-to-large-language-models.pdf"
    text = read_data_from_pdf(pdf_path)
    chunks = get_text_chunks(text)
    points = get_points(client, chunks)
    insert_points(connection, points)
    query = "What is Hallucication in the context of LLMs?"
    context = search_context(client, connection, query)
    print(context)
    answer = generate_openai(client, query, context)
    print(f"Answer: {answer}")
    
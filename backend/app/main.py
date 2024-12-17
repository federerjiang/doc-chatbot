from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import AsyncQdrantClient
from elasticsearch import Elasticsearch

# from langchain_community.retrievers import ElasticSearchBM25Retriever

import os 
import logging
import spacy 

from .es_util import ElasticSearchBM25Retriever
from .rrf_util import reciprocal_rank_fusion, rerank
from .llm_util import client as client_openai
from .qdrant_util import qdrant_search_context, qdrant_get_points, qdrant_insert_points
from .index import read_data_from_pdf, get_text_chunks 


# 设置基本的配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IS_OPENAI_EMBEDDING = True
IS_OPENAI_MODEL = True
COLLECTION_NAME = "knowledge_base_3"

OLLAMA_HOST = "ollama"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

OLLAMA_EMBED_HOST = "ollama-embedding"
OLLAMA_EMBED_PORT = 11435
OLLAMA_EMBED_BASE_URL = f"http://{OLLAMA_EMBED_HOST}:{OLLAMA_EMBED_PORT}"

elasticsearch_url = "http://es01:9200"
es_client = Elasticsearch(elasticsearch_url)
retriever = ElasticSearchBM25Retriever(client=es_client, index_name=COLLECTION_NAME)
nlp = spacy.load("ja_ginza")

connection = AsyncQdrantClient("http://qdrant:6333")

def extract_keywords(query):
    doc = nlp(query)
    nouns = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
    return nouns

llm = Ollama(
    # model="federer/lightblue-llama3", 
    model="federer/elyza-japanese",
    temperature=0, 
    base_url=OLLAMA_BASE_URL, 
    system="""
    あなたは誠実で優秀な日本人のアシスタントです。以下のコンテキスト情報を元に質問に回答してください。""",
)
if IS_OPENAI_EMBEDDING:
    embed_client = client_openai
else:
    embed_client = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_EMBED_BASE_URL)

app = FastAPI()

class Message(BaseModel):
    text: str

origins = [
    "*",
    "http://localhost:3000/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

async def startup_event():
    # Insert data into the collection
    try:
        await connection.http.healthcheck() 
    except Exception as e:
        connection = AsyncQdrantClient("http://qdrant:6333")
    if not await connection.collection_exists(COLLECTION_NAME):
        points = []
        for pdf_file in os.listdir("assets"):
            pdf_path = f"assets/{pdf_file}"
            text = read_data_from_pdf(pdf_path)
            chunks = get_text_chunks(text)
            retriever.add_texts(chunks)
            points += qdrant_get_points(embed_client, chunks, is_openai=IS_OPENAI_EMBEDDING)
        await qdrant_insert_points(connection, points, collection_name=COLLECTION_NAME, is_openai=IS_OPENAI_EMBEDDING)

@app.on_event("startup")
async def startup():
    await startup_event()
    print("Startup complete.")

@app.get("/messages")
async def get_initial_message():
    return {"message": "こんにちは!"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        while True:
            text_data = await websocket.receive_text()
            query = text_data
            logger.info(f"Received data: {text_data}")
            try:
                await connection.http.healthcheck() 
            except Exception as e:
                logger.error(f"Healthcheck failed: {e}")
                connection = AsyncQdrantClient("http://qdrant:6333")
                logger.info("Reconnected to Qdrant")
                
            qdrant_contexts = await qdrant_search_context(embed_client, connection, query, collection_name=COLLECTION_NAME, is_openai=IS_OPENAI_EMBEDDING)
            logger.info(f"Context from qdrant: {qdrant_contexts}")
            keywords = extract_keywords(query)
            keywords_str = ",".join(keywords)
            es_contexts = [doc.page_content for doc in await retriever.ainvoke(keywords_str)]
            logger.info(f"Context from elasticsearch: {es_contexts}")
            # reranked_contexts = reciprocal_rank_fusion([qdrant_contexts, es_contexts])
            reranked_contexts = reciprocal_rank_fusion(qdrant_contexts, es_contexts, top_k=5, qdrant_k=30, es_k=60)
            reranked_contexts = rerank(query, reranked_contexts)
            context = " ".join(reranked_contexts)
            logger.info(f"Reranked context: {context}")
            if IS_OPENAI_MODEL:
                # user_prompt = f""" You need to answer the question in the sentences as same as in the context.
                #     Given below is the context and question of the user.
                    
                #     context = {context}
                    
                #     question = {query}
                    
                #     if the answer is not in the context or context is empty, answer "何を聞いているのか分かりません".
                #     answer in the same language as in the question.    
                #     """
                user_prompt = f"""Use the following pieces of context to answer the question at the end.
                                If you don't know the answer, just say that you don't know, don't try to make up an answer.

                                {context}

                                Question: {query}
                                answer in Japanese with detail."""
            else:
                user_prompt = f"""
                    もし以下のコンテキスト内に答えがない場合は、「何を聞いているのか分かりません」と答えてください。

                    {context}
                
                    {query}? 質問に直接答えてください。
                """

            logger.info(f"Generated prompt: {user_prompt}")
            if IS_OPENAI_MODEL:
                response = client_openai.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {'role': 'user', 'content': user_prompt},
                    ],
                    temperature=0,
                    stream=True  # this time, we set stream=True
                )
                for chunk in response:
                    logger.info(chunk)
                    logger.info(chunk.choices[0].delta.content)
                    if chunk.choices[0].delta.content:
                        await websocket.send_text(chunk.choices[0].delta.content)
                        logger.info(f"Sent chunk: {chunk.choices[0].delta.content}")
                await websocket.send_text('**|||END|||**')
            else:
                for chunks in llm.stream(user_prompt):
                    await websocket.send_text(chunks)
                    logger.info(f"Sent chunk: {chunks}")
                await websocket.send_text('**|||END|||**')
            logger.info("Sent end marker")
    except Exception as e:
        logger.exception("An error occurred in websocket endpoint")
        await websocket.close(code=1011)  # WebSocket internal error
        logger.info("WebSocket connection closed due to error")
    finally:
        await websocket.close()
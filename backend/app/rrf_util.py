import logging
import requests

# 设置基本的配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(qdrant_results: list[str], es_results: list[str], top_k=5, qdrant_k=30, es_k=60):
    """Rerank docs (Reciprocal Rank Fusion)

    Args:
        results (list[list]): retrieved documents
        k (int, optional): parameter k for RRF. Defaults to 60.

    Returns:
        ranked_results: list of documents reranked by RRF
    """

    fused_scores = {}
    for rank, doc in enumerate(qdrant_results):
        if doc not in fused_scores:
            fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + qdrant_k)
    for rank, doc in enumerate(es_results):
        if doc not in fused_scores:
            fused_scores[doc] = 0
        fused_scores[doc] += 1 / (rank + es_k)

    reranked_results = [
        (doc, score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # for TEST (print reranked documentsand scores)
    logger.info(f"RRF Ranked Documents: {len(reranked_results)}", )
    if top_k > len(reranked_results):
        top_k = len(reranked_results)
    for result in reranked_results[:top_k]:
        logger.info(f'Docs: {result[0]}')
        logger.info(f'RRF score: {result[1]}')

    # return only documents
    return [x[0] for x in reranked_results[:top_k]]


def rerank(query, results, top_k=3):
    # API的URL
    url = "http://rerank:8001/score/"
    doc_scores = {}
    for doc in results:
        data = {
            "sentence1": query,
            "sentence2": doc,
        }
        # 发送POST请求
        response = requests.post(url, json=data)
        # 检查响应状态码
        if response.status_code == 200:
        # 获取并打印得分
            score = response.json()['score']
            logger.info(f"Similarity score: {score}")
        else:
            score = 0
            logger.info(f"Failed to retrieve score, status code: {response.status_code}")
        doc_scores[doc] = score
    
    reranked_results = [
        (doc, score)
        for doc, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    for result in reranked_results:
        logger.info(f'Docs: {result[0]}')
        logger.info(f'Rerank score: {result[1]}')
        
    final_results = []
    for result in reranked_results:
        if result[1] >= 0.01:
            final_results.append(result[0])
    logger.info(f"Final Reranked Documents: {len(final_results)}")
    logger.info(f"Final Reranked Documents: {final_results}")
    if top_k > len(final_results):
        top_k = len(final_results)
    if top_k == 0:
        return [] 
    else:
        return final_results[:top_k]
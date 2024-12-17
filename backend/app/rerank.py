import requests

# API的URL
url = "http://rerank:8001/score/"

# 要评分的两个句子
data = {
    "sentence1": "感動的な映画について",
    "sentence2": "深いテーマを持ちながらも、観る人の心を揺さぶる名作。"
}

# 发送POST请求
response = requests.post(url, json=data)

# 检查响应状态码
if response.status_code == 200:
    # 获取并打印得分
    score = response.json()['score']
    print(f"Similarity score: {score}")
else:
    print(f"Failed to retrieve score, status code: {response.status_code}")

import requests
import os
import numpy as np


def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    :param vec1: 第一个向量
    :param vec2: 第二个向量
    :return: 余弦相似度值
    """
    # 将列表转换为NumPy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # 计算点积
    dot_product = np.dot(vec1, vec2)

    # 计算向量的模长
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 计算余弦相似度
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)

    return cosine_sim

# 定义 API URL
url = "https://api.baichuan-ai.com/v1/embeddings"

# 从环境变量中获取 API 密钥
api_key = ''

# 定义请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 定义请求数据
data = {
    "model": "Baichuan-Text-Embedding",
    "input": ["分布式光伏","太阳能发电"]
}

# 发送 POST 请求并处理可能的异常
try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # 检查请求是否成功
except requests.exceptions.HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')  # Python 3.6
except Exception as err:
    print(f'An error occurred: {err}')
else:
    print('Success!')
    # 打印响应的 JSON 数据
    print(response.json())
    embedding1 = response.json()['data'][0]['embedding']
    embedding2 = response.json()['data'][1]['embedding']

    # 计算余弦相似度
    similarity = cosine_similarity(embedding1, embedding2)

    print(f"余弦相似度: {similarity}")

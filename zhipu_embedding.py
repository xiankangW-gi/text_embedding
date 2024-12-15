import csv
import numpy as np
from zhipuai import ZhipuAI


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


def process_csv_embeddings(input_csv_path, output_csv_path, api_key):
    """
    读取CSV文件，获取嵌入向量并重新计算余弦相似度
    :param input_csv_path: 输入CSV文件路径
    :param output_csv_path: 输出CSV文件路径
    :param api_key: Zhipuai API密钥
    """
    # 读取原始CSV文件
    words = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行（如果有的话）
        for row in reader:
            words.extend([row[0], row[1]])

    # 去重并获取唯一词语
    unique_words = list(set(words))

    # 使用Zhipuai获取嵌入向量（分批处理）
    client = ZhipuAI(api_key=api_key)
    embeddings_dict = {}

    # 分批处理，每批最多64个词
    for i in range(0, len(unique_words), 64):
        batch = unique_words[i:i + 64]

        response = client.embeddings.create(
            model="embedding-3",
            input=batch,
            dimensions=512
        )

        # 更新嵌入向量字典 - 根据索引匹配原始词语
        for idx, embedding_item in enumerate(response.data):
            word = batch[idx]
            embeddings_dict[word] = embedding_item.embedding

    # 重新计算余弦相似度
    output_data = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 保存原始标题

        for row in reader:
            word1, word2, original_similarity = row

            # 获取对应的嵌入向量
            vec1 = embeddings_dict[word1]
            vec2 = embeddings_dict[word2]

            # 计算新的余弦相似度
            new_similarity = cosine_similarity(vec1, vec2)

            # 保存结果
            output_data.append([
                word1, word2,
                original_similarity,
                str(new_similarity)
            ])

    # 写入输出CSV文件
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers + ['New Similarity'])
        writer.writerows(output_data)

    print(f"处理完成。结果已保存到 {output_csv_path}")


# 使用示例
input_file = 'D:/similar_words(1).csv'
output_file = 'output.csv'
api_key = '5317479a18b33cf39f3883277eba54fe.Nncl4jLCUaY1C7dZ'  # 请替换为你的实际API密钥

process_csv_embeddings(input_file, output_file, api_key)
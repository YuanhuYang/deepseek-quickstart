# -*- coding: utf-8 -*-

import os
from openai import OpenAI
from pymilvus import model as milvus_model
from pymilvus import MilvusClient
from tqdm import tqdm
import json

# 1. 环境变量与 API Key
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")

# 2. 加载民法典文档数据
mfd_path = "deepseek/api/mfd.md"
with open(mfd_path, "r", encoding="utf-8") as file:
    mfd_text = file.read()
text_lines = mfd_text.split("# ")

print(f"文档分段数: {len(text_lines)}")
print("示例分段:", text_lines[:5])

# 3. 初始化 DeepSeek 客户端
deepseek_client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
)

# 4. 初始化 embedding 模型
embedding_model = milvus_model.DefaultEmbeddingFunction()

# 5. 测试 embedding 维度
test_embedding = embedding_model.encode_queries(["This is a test"])[0]
embedding_dim = len(test_embedding)
print("Embedding 维度:", embedding_dim)
print("Embedding 前10维:", test_embedding[:10])

# 6. 初始化 Milvus Lite
os.environ["TOKENIZERS_PARALLELISM"] = "false"
milvus_client = MilvusClient(uri="./milvus_mfd.db")
collection_name = "mfd_knowledge_base"

# 7. 清理旧 collection
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

# 8. 创建新 collection
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # 内积距离
    consistency_level="Strong",
)

# 9. 文档嵌入并插入 Milvus
data = []
doc_embeddings = embedding_model.encode_documents(text_lines)
for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": doc_embeddings[i], "text": line})

milvus_client.insert(collection_name=collection_name, data=data)
print(f"已插入 {len(data)} 条数据到 Milvus。")

# 10. 提出问题并检索
questions = [
    "民法典中关于合同的基本原则是什么？",
    "民法典中如何定义自然人？"
]

for question in questions:
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=embedding_model.encode_queries([question]),
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    readable_results = [
        f"段落: {line_with_distance[0]}\n相似度: {line_with_distance[1]:.4f}"
        for line_with_distance in retrieved_lines_with_distances
    ]
    print(f"问题: {question}")
    print("检索结果:")
    print("\n".join(readable_results))

    # 11. 构建 RAG 上下文
    context = "\n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distances])

    # 12. 构造提示词
    SYSTEM_PROMPT = """Human: 你是一个 AI 助手。你能够从提供的上下文段落片段中找到问题的答案。"""
    USER_PROMPT = f"""请使用以下用 <context> 标签括起来的信息片段来回答用 <question> 标签括起来的问题。最后追加原始回答的中文翻译，并用 <translated>和</translated> 标签标注。
<context>
{context}
</context>
<question>
{question}
</question>
<translated>
</translated>
"""

    # 13. 调用 LLM 生成答案
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    print("RAG 回答：")
    print(response.choices[0].message.content)

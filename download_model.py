from huggingface_hub import snapshot_download

# 下载模型到本地
# 这里使用了 Hugging Face Hub 的 snapshot_download 函数来下载模型到指定的本地目录。


# moka-ai/m3e-large 是一个大型的多模态模型，适用于文本和图像的处理任务。它可以用于各种自然语言处理和计算机视觉任务，如文本分类、图像识别等。
snapshot_download(repo_id="moka-ai/m3e-large", local_dir="./pre_train_model/m3e-large")

# BAAAI/bge-m3 是一个文本编码器模型，适用于文本相似度计算和文本检索任务。它可以将文本转换为固定长度的向量表示，便于进行相似度计算和信息检索。
snapshot_download(repo_id="BAAI/bge-m3", local_dir="./pre_train_model/bge-m3")

# InfiniFlow/bce-reranker-base_v1 是一个基于BCE（Binary Cross-Entropy）损失函数的重排序模型，适用于信息检索和自然语言处理任务。它可以用于对候选答案进行重新排序，以提高检索结果的相关性和准确性。
snapshot_download(repo_id="InfiniFlow/bce-reranker-base_v1", local_dir="./pre_train_model/bce-reranker-base_v1")

# shibing624/text2vec-base-chinese 是一个中文文本向量化模型，适用于中文文本的表示和相似度计算任务。它可以将中文文本转换为固定长度的向量表示，便于进行文本相似度计算和信息检索。
snapshot_download(repo_id="shibing624/text2vec-base-chinese", local_dir="./pre_train_model/text2vec-base-chinese")


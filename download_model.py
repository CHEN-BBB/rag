from huggingface_hub import snapshot_download
snapshot_download(repo_id="moka-ai/m3e-large", local_dir="./pre_train_model/m3e-large")
snapshot_download(repo_id="BAAI/bge-m3", local_dir="./pre_train_model/bge-m3")
snapshot_download(repo_id="InfiniFlow/bce-reranker-base_v1", local_dir="./pre_train_model/bce-reranker-base_v1")
snapshot_download(repo_id="shibing624/text2vec-base-chinese", local_dir="./pre_train_model/text2vec-base-chinese")


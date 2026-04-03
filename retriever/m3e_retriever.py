from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
import torch
import os

script_dir = os.path.dirname(__file__)


# Dense语义召回M3E
class M3eRetriever(object):
    def __init__(self, embeddings_model_path=None, data_path=None, vector_path=None, pdf_path=None):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model_path,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 64}
        )
        # 如果没有向量数据缓存，则从pdf解析后的文本数据或者pdf文件中处理得到文本数据，并构建faiss索引进行向量化存储；如果有向量数据缓存，则直接加载faiss索引进行向量化存储
        if vector_path is None:
            docs = []
            if data_path is not None:
                # 从pdf解析后的文本数据中处理得到文档对象列表，并构建faiss索引进行向量化存储
                with open(data_path, "r", encoding="utf-8") as file:
                    docs = self.data_process(file)
            elif pdf_path is not None:
                dp = DataProcess(pdf_path)
                # pdf解析方法一，基于块的解析方法，块的定义是相邻文本行的字体大小相同，且块内文本行的字体大小与块外文本行的字体大小不同
                dp.ParseBlock(max_seq=1024)
                dp.ParseBlock(max_seq=512)
                # pdf解析方法二，基于滑窗的解析方法，滑窗的定义是以固定长度的文本块为单位，按照一定的步长进行滑动，直到覆盖整个文档
                dp.ParseAllPage(max_seq=256)
                dp.ParseAllPage(max_seq=512)
                # pdf解析方法三，基于规则的解析方法，规则的定义是以句号为分割点，将文本分割成若干个句子，然后按照一定的重叠长度进行合并，直到覆盖整个文档
                dp.ParseOnePageWithRule(max_seq=256)
                dp.ParseOnePageWithRule(max_seq=512)
                print("m3e pdf_parse is ok")
                # 对pdf解析后的文本数据进行处理，得到文档对象列表，并构建faiss索引进行向量化存储
                docs = self.data_process(dp.data)
            # 构建faiss索引进行向量化存储
            self.vector_store = FAISS.from_documents(docs, self.embeddings)
            self.vector_store.save_local(os.path.normpath(os.path.join(script_dir, "../vector_db/faiss_m3e_index")))
            print("m3e faiss vector_db is ok")
        else:
            # 加载faiss索引进行向量化存储
            self.vector_store = FAISS.load_local(vector_path, self.embeddings, allow_dangerous_deserialization=True)
        # 释放显存占用，因为后续还会加载bge_retriever，而m3e和bge的文本嵌入模型都是基于transformer大模型的，显存占用较大，所以在加载完m3e_retriever后，先释放显存占用，再
        del self.embeddings
        torch.cuda.empty_cache()

    # 对pdf解析后的文本数据进行处理
    def data_process(self, data):
        docs = []
        # 对pdf解析后的文本数据进行处理，得到文档对象列表，文档对象包含文本内容和元数据信息，元数据信息包含文本在pdf中的位置等信息
        for idx, line in enumerate(data):
            # 对文本内容进行预处理，去除换行符和制表符等特殊字符，然后将文本内容和元数据信息封装成文档对象，并添加到文档对象列表中
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        return docs

    # 获取top_K分数最高的文档块
    def GetTopK(self, query, k):
        context = self.vector_store.similarity_search_with_score(query, k=k)
        return context

    def GetvectorStore(self):
        return self.vector_store


if __name__ == "__main__":
    embeddings_model_path = "../pre_train_model/m3e-large"
    data_path = "../all_text.txt"
    vector_path = "../vector_db/faiss_m3e_index"
    pdf_path = "../data/car_user_manual.pdf"

    # m3e_retriever = M3eRetriever(embeddings_model_path, pdf_path)
    m3e_retriever = M3eRetriever(embeddings_model_path, data_path, vector_path)
    m3e_ans = m3e_retriever.GetTopK("座椅加热", 3)
    print(m3e_ans)

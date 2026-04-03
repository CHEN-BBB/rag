import argparse
from config import M3E_embeddings_model_path, BGE_embeddings_model_path, SimModel_path
from generate_answer import question_test
from test_score import test_metrics


parser = argparse.ArgumentParser(description='Intelligent Cabin Automotive Knowledge Q&A System')
# 选择大语言模型
parser.add_argument('--llm_name', default='qwen2', type=str,
                    choices=['qwen2', 'baichuan2', 'chatglm3', "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3-9B"],
                    help='Select the Large Language Model for Generating Responses')
# 选择重排序模型
parser.add_argument('--reranker_name', default='bce', type=str, choices=['bce', 'bge'],
                    help='Select the reranking model used for reordering the retrieved documents')
# 选择文本嵌入模型
parser.add_argument('--m3e_embeddings_model', default=M3E_embeddings_model_path, type=str,
                    help='The path to the text embedding model for recall used by M3E')
# 选择文本嵌入模型
parser.add_argument('--bge_embeddings_model', default=BGE_embeddings_model_path, type=str,
                    help='The path to the text embedding model for recall used by BGE')
# 选择是否进行提示词优化
parser.add_argument('--prompt_enhance', default=True, type=str,
                    help='Choose to optimize the prompt')
# 选择单路径召回的最大文本长度
parser.add_argument('--single_max_length', default=4000, type=int,
                    help='The maximum text length for single-path recall')
# 选择单路径召回的最大召回数量
parser.add_argument('--single_top_k', default=6, type=int,
                    help='The maximum number of retrievals for single-path recall')
# 选择多路径召回的最大文本长度
parser.add_argument('--mutil_max_length', default=4000, type=int,
                    help='The maximum text length for multi-path recall')
# 选择多路径召回的最大召回数量
parser.add_argument('--mutil_top_k', default=6, type=int,
                    help='The maximum number of retrievals for multi-path recall')
# 选择PDF文件的路径
parser.add_argument('--pdf_path', default="./data/car_user_manual.pdf", type=str,
                    help='The path to the PDF file')
# 选择测试集的路径
parser.add_argument('--test_path', default="./data/test_question.json", type=str,
                    help='The path to the test dataset')
# 选择预测结果的存储路径
parser.add_argument('--predict_path', default="./data/result.json", type=str,
                    help='The storage path for the prediction results')
# 选择标准答案数据集的路径
parser.add_argument('--gold_path', default="./data/gold_result.json", type=str,
                    help='The path to the standard answer dataset')
# 选择相似度计算模型的路径
parser.add_argument('--simModel_path', default=SimModel_path, type=str,
                    help='The similarity model used for calculating scores')
# 选择评测指标数据的存储路径
parser.add_argument('--metric_path', default="./data/metrics.json", type=str,
                    help='The storage path for evaluation metric data')
# 选择PDF文件处理后的缓存数据路径
parser.add_argument('--data_path', default="./all_text.txt", type=str,
                    help='The storage path after parsing the PDF file')
# 选择m3e召回的向量数据缓存路径
parser.add_argument('--m3e_vector_path', default="./vector_db/faiss_m3e_index", type=str,
                    help='The vector database based on M3E recall')
# 选择bge召回的向量数据缓存路径
parser.add_argument('--bge_vector_path', default="./vector_db/faiss_bge_index", type=str,
                    help='The vector database based on BGE recall')

args = parser.parse_args()


if __name__ == '__main__':

    # 基于rag的测试集预测
    question_test(
        model_name=args.llm_name,
        reranker_name=args.reranker_name,
        m3e_embeddings_model_path=args.m3e_embeddings_model,
        bge_embeddings_model_path=args.bge_embeddings_model,
        test_path=args.test_path,
        output_path=args.predict_path,
        prompt_enhance=args.prompt_enhance,
        single_max_length=args.single_max_length,
        single_top_k=args.single_top_k,
        mutil_max_length=args.mutil_max_length,
        mutil_top_k=args.mutil_top_k,
        pdf_path=args.pdf_path,
        # data_path=args.data_path,                 # pdf文件处理后的缓存数据
        # m3e_vector_path=args.m3e_vector_path,     # m3e召回的向量数据缓存
        # bge_vector_path=args.bge_vector_path,     # bge召回的向量数据缓存
    )

    # 测试集预测结果的综合得分计算
    test_metrics(
        gold_path=args.gold_path,
        predict_path=args.predict_path,
        metric_path=args.metric_path,
        simModel_path=args.simModel_path,
    )

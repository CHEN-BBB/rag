import json
import time
from tqdm import tqdm
from hf_model import ChatLLM
from huggingface_proxy import HFProxy
from rerank_model import reRankLLM
from retriever.m3e_retriever import M3eRetriever
from retriever.bge_retriever import BgeRetriever
from retriever.bm25_retriever import Bm25Retriever
from retriever.tfidf_retriever import TfidfRetriever


# 获取输入模版,输入模版包含已知信息和问题，其中已知信息是基于召回的文档构造的，问题是用户输入的问题，输出是一个字符串，字符串是一个提示模版，提示模版包含已知信息和问题，提示模版用于大模型生成答案
def get_prompt_template(docs, query):
    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。]如果无法从中得到答案，请说"无答案" ，不允许在答案中添加编造成分，答案请使用中文。\n\
已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:\n{retriever_text}\n问题: {question}\n回答:""".format(
        retriever_text=docs, question=query)
    return prompt_template


# 基于召回的文档构造一个提示模版，并返回召回文档中与query的最小距离分数,m3e和bge召回的文档都有距离分数，bm25和tfidf没有距离分数，直接返回提示模版
def get_emb_docs(m3e_context, query, max_length=4000, top_k=6):
    m3e_min_score = 0.0
    if (len(m3e_context) > 0):
        # 获取召回文档中与query的最小距离分数
        m3e_min_score = m3e_context[0][1]
    cnt = 0
    emb_ans = ""
    for doc, score in m3e_context:
        cnt = cnt + 1
        # 如果召回文档中与query的最小距离分数大于max_length，说明召回文档与query的相关性较低，可能无法提供有效的信息，此时可以选择不将召回文档作为已知信息输入到提示模版中，直接返回一个提示模版，提示模版中不包含已知信息，只包含问题，提示模版用于大模型生成答案
        if (len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
        if (cnt > top_k):
            break
    m3e_prompt_template = get_prompt_template(emb_ans, query)
    return m3e_prompt_template, m3e_min_score


# 基于召回的文档构造一个提示模版，bm25和tfidf没有距离分数，直接返回提示模版
def get_distribute_docs(bm25_context, query, max_length=4000, top_k=6):
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if (len(bm25_ans + doc.page_content) > max_length):
            break
        bm25_ans = bm25_ans + doc.page_content
        if (cnt > top_k):
            break
    bm25_prompt_template = get_prompt_template(bm25_ans, query)
    return bm25_prompt_template


# 基于多路召回的文档构造一个提示模版,并返回重排后前top_k条文档构成的提示模版，m3e和bge召回的文档都有距离分数，bm25和tfidf没有距离分数，直接返回提示模版
def get_emb_distribute_rerank(rerank_model, m3e_context, bge_context, bm25_context, tfidf_context, query,
                              max_length=4000, top_k=6):
    items = []
    for doc, score in m3e_context:
        items.append(doc)
    for doc, score in bge_context:
        items.append(doc)
    items.extend(bm25_context) # bm25和tfidf没有距离分数，直接将召回文档添加到items列表中
    items.extend(tfidf_context)
    # 基于多路召回的文档构造一个提示模版,并返回重排后前top_k条文档构成的提示模版
    rerank_ans = rerank_model.predict(query, items)
    rerank_ans_k = rerank_ans[:top_k]
    rerank_text = ""
    for doc in rerank_ans_k:
        if len(rerank_text + doc.page_content) > max_length:
            break
        rerank_text = rerank_text + doc.page_content
    mutil_rerank_prompt_template = get_prompt_template(rerank_text, query)
    return mutil_rerank_prompt_template


# 基于多路召回的文档构造一个提示模版,并返回前top_k条文档构成的提示模版，m3e和bge召回的文档都有距离分数，bm25和tfidf没有距离分数，直接返回提示模版
def get_emb_distribute_no_rerank(m3e_context, bge_context, bm25_context, tfidf_context, query, mutil_max_length=4000, mutil_top_k=6):
    items = []
    for doc, score in m3e_context:
        items.append(doc)
    for doc, score in bge_context:
        items.append(doc)
    items.extend(bm25_context) # bm25和tfidf没有距离分数，直接将召回文档添加到items列表中
    items.extend(tfidf_context)
    # 去重（按 page_content）
    seen = set()
    unique_docs = []
    for doc in items:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    # 截取 top_k，拼成 prompt
    merge_text = ""
    for doc in unique_docs[:mutil_top_k]:
        if len(merge_text + doc.page_content) > mutil_max_length:
            break
        merge_text += doc.page_content
    no_rerank_prompt_template = get_prompt_template(merge_text, query)
    return no_rerank_prompt_template


# 如果输出内容超长（>500字）或包含Prompt关键词，认为是输出异常
def clean_answer(answer):
    if len(answer) > 500 or "基于以下已知信息" in answer or "You are a helpful assistant" in answer:
        return "无答案"
    return answer


# 对测试数据集进行rag评测
def question_test(model_name=None, reranker_name=None, use_rerank=True, m3e_embeddings_model_path=None, bge_embeddings_model_path=None,
                  pdf_path=None, test_path=None, output_path=None, data_path=None, m3e_vector_path=None, prompt_enhance=True,
                  bge_vector_path=None, single_max_length=4000, single_top_k=6, mutil_max_length=4000, mutil_top_k=6):
    start = time.time()
    # 初始化检索模型
    m3e_retriever = M3eRetriever(m3e_embeddings_model_path, data_path, m3e_vector_path, pdf_path)
    print("m3e_retriever load ok")
    bge_retriever = BgeRetriever(bge_embeddings_model_path, data_path, bge_vector_path, pdf_path)
    print("bge_retriever load ok")
    bm25 = Bm25Retriever(data_path, pdf_path)
    print("bm25 load ok")
    tfidf = TfidfRetriever(data_path, pdf_path)
    print("tf-idf load ok")

    # LLM大模型
    if model_name and "Qwen" in model_name:
        llm = HFProxy(model=model_name)
    else:
        llm = ChatLLM(model_name)
    print("llm load ok")

    # reRank模型
    if (use_rerank):
        rerank = reRankLLM(reranker_name)
        print("rerank model load ok")

    # 对每一条测试问题，做答案生成处理
    if test_path is None or output_path is None:
        raise ValueError("test_path and output_path must be provided")
    with open(test_path, "r", encoding="utf-8") as file:
        jdata = json.loads(file.read())
        print(len(jdata))
        for idx, line in tqdm(enumerate(jdata), total=len(jdata)):
            query = line["question"]
            retriever_query = query
            if prompt_enhance:
                prompt = "请简洁地回答下面的问题，只需要输出答案，不要重复问题，不允许在答案中添加编造成分，注意输出内容控制在16个字以内。\n问题：" + query
                answer = llm.infer([prompt])
                retriever_query = query + answer[0]

            # 召回文档
            m3e_context = m3e_retriever.GetTopK(retriever_query, 15)
            bge_context = bge_retriever.GetTopK(retriever_query, 15)
            bm25_context = bm25.GetBM25TopK(retriever_query, 15)
            tfidf_context = tfidf.GetBM25TopK(retriever_query, 15)

            # 重排文档
            m3e_inputs, m3e_min_score = get_emb_docs(m3e_context, query, max_length=single_max_length,
                                                     top_k=single_top_k)
            bge_inputs, bge_min_score = get_emb_docs(bge_context, query, max_length=single_max_length,
                                                     top_k=single_top_k)
            bm25_inputs = get_distribute_docs(bm25_context, query, max_length=single_max_length, top_k=single_top_k)
            tfidf_inputs = get_distribute_docs(tfidf_context, query, max_length=single_max_length, top_k=single_top_k)
            if (use_rerank):
                mutil_rerank_inputs = get_emb_distribute_rerank(rerank, m3e_context, bge_context, bm25_context,
                                                            tfidf_context, query, max_length=mutil_max_length,
                                                            top_k=mutil_top_k)
            else:
                mutil_rerank_inputs = get_emb_distribute_no_rerank(m3e_context, bge_context, bm25_context,
                                                            tfidf_context, query, mutil_max_length, mutil_top_k)

            # 基于同一个问题构建一组batch
            batch_input = []
            batch_input.append(mutil_rerank_inputs)
            batch_input.append(m3e_inputs)
            batch_input.append(bge_inputs)
            batch_input.append(bm25_inputs)
            batch_input.append(tfidf_inputs)
            # 执行batch推理
            batch_output = llm.infer(batch_input)
            line["answer_1"] = clean_answer(batch_output[0])     # 多路召回重排序后的结果
            line["answer_2"] = clean_answer(batch_output[1])     # m3e召回的结果
            line["answer_3"] = clean_answer(batch_output[2])     # bge召回的结果
            line["answer_4"] = clean_answer(batch_output[3])     # bm25召回的结果
            line["answer_5"] = clean_answer(batch_output[4])     # tfidf召回结果

            # 如果m3e或bge检索跟query的距离高于500，输出无答案
            if m3e_min_score > 500:
                line["answer_6"] = "无答案"
            if bge_min_score > 500:
                line["answer_7"] = "无答案"

        # 保存结果
        json.dump(jdata, open(output_path, "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()

        print("cost time: " + str(int(end - start) / 60) + "minutes")

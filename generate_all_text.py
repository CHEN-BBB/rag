from pdf_parse import DataProcess

# 初始化 PDF 解析器
dp = DataProcess("./data/car_user_manual.pdf")

# 执行多种解析策略（与 retriever 中相同）
dp.ParseBlock(max_seq=1024)
dp.ParseBlock(max_seq=512)
dp.ParseAllPage(max_seq=256)
dp.ParseAllPage(max_seq=512)
dp.ParseOnePageWithRule(max_seq=256)
dp.ParseOnePageWithRule(max_seq=512)

print("PDF 解析完成，开始写入 all_text.txt")

# 将解析结果写入 all_text.txt
with open("./all_text.txt", "w", encoding="utf-8") as f:
    for line in dp.data:
        f.write(line + "\n")

print("all_text.txt 生成完成！")
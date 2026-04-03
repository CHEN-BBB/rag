import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()


class HFProxy():
    def __init__(self, model="Qwen/Qwen2-7B-Instruct", temperature=0.1):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )

    def get_response(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"请求失败: {str(e)}"

    def infer(self, prompts):
        # 使用线程池并行处理
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.get_response, prompts))
        return results


if __name__ == "__main__":
    prompts = [
        "你好",
        "你会干什么",
        "推荐5本人工智能入门书籍"
    ]
    llm = HFProxy()
    results = llm.infer(prompts)
    print(results)

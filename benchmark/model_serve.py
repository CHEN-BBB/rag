import torch
from typing import List, Dict
from collections import defaultdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 定义请求体的数据结构
class ChatRequest(BaseModel):
    message: List[Dict[str, str]]


# 全局变量
model_path = "../models/Qwen2-7B-Instruct"
model = None
tokenizer = None


# 在服务启动时加载模型
@app.on_event("startup")
async def load_model():
    global model, tokenizer, model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()


# 处理聊天请求
@app.post("/qwen")
async def qwen(chat_request: ChatRequest):
    message = chat_request.message
    result = defaultdict(str)

    # 使用模型生成回复
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # 去除输入部分，只保留输出
    if prompt in response:
        response = response[len(prompt):]

    result["role"] = "assistant"
    result["content"] = response.strip()
    return result


# 启动服务
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model_serve:app", host="127.0.0.1", port=8000, reload=True)

import torch
from config import *
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatLLM(object):
    def __init__(self, model_name):
        self.model_name = model_name
        if self.model_name == "qwen2":
            self.model_path = Qwen2_path
        if self.model_name == "baichuan2":
            self.model_path = Baichuan_path
        if self.model_name == "chatglm3":
            self.model_path = ChatGLM_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side='left', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # HuggingFace 生成配置
        self.generation_config = GenerationConfig.from_pretrained(self.model_path, trust_remote_code=True)

    # 批量推理，输入一个batch，返回一个batch的答案
    def infer(self, prompts):
        batch_text = []
        for q in prompts:
            if self.model_name == "qwen2":
                text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            if self.model_name == "baichuan2":
                text = f"<reserved_106>{q}<reserved_107>"
            if self.model_name == "chatglm3":
                text = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{q}\n<|assistant|>\n"
            batch_text.append(text)

        batch_response = []
        for text in batch_text:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=2000,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                    repetition_penalty=1.05,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 去除输入部分，只保留输出
            output_str = self.tokenizer.decode(output[0], skip_special_tokens=True)
            if text in output_str:
                output_str = output_str[len(text):]
            
            batch_response.append(output_str.strip())

        torch_gc()
        return batch_response


if __name__ == "__main__":
    model_name = "qwen2"
    start = time.time()
    llm = ChatLLM(model_name)
    test = ["你好", "吉利汽车语音组手唤醒", "自动驾驶功能介绍"]
    generated_text = llm.infer(test)
    print(generated_text)
    end = time.time()
    print("cost time: " + str((end - start) / 60) + "minutes")

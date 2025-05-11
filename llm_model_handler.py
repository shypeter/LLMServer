from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class LLMModelHandler:
    def __init__(self):
        self.model_name = "openai/whisper-small"
        self.token = ""
        try:
            # Print GPU information
            if torch.cuda.is_available():
                print(f"GPU Model: {torch.cuda.get_device_name(0)}")
                print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                torch.cuda.empty_cache()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, self.token)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.token,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            print(f"模型 {self.model_name} 已成功載入")
        except Exception as e:
            print(f"載入模型 {self.model_name} 時發生錯誤: {e}")
            self.llm = None
            self.tokenizer = None

    def generate_answer(self, query: str, context: str = "", max_tokens: int = 512) -> str:
        if not self.llm or not self.tokenizer:
            return "LLM模型未初始化或載入失敗"

        if context:
            prompt = f"""以下是一些相關文件的內容:

{context}

基於以上信息，用繁體中文台灣用語回答:
{query}

回答:"""
        else:
            prompt = f"""用繁體中文台灣用語回答:
{query}

回答:"""
        
        try:
            # 將提示轉換為輸入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
            
            # 生成回答
            with torch.no_grad():
                output = self.llm.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            
            # 解碼生成的回答
            response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            return response.strip()
        except Exception as e:
            print(f"生成回答時出錯: {str(e)}")
            return f"生成過程中出錯: {str(e)}"
#
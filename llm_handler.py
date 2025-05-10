from llama_cpp import Llama
import os

class LLMHandler:
    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            self.llm = None
        else:
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_gpu_layers=99999,  # 使用最大值來嘗試將所有層都放在 GPU 上
                    main_gpu=0,          # 主 GPU 索引
                    tensor_split=None,    # 若有多個 GPU，可以設定張量分割，例如 [0.5, 0.5]
                    n_threads=4
                )
                print(f"模型 {model_path} 已成功載入")
            except Exception as e:
                print(f"載入模型 {model_path} 時發生錯誤: {e}")
                self.llm = None

    def generate_answer(self, query: str, context: str="", max_tokens: int=512) -> str:
        if not self.llm:
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
            # 生成回答
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                stop=["問題:", "問:"],
                echo=False
            )
            
            if output and "choices" in output and len(output["choices"]) > 0:
                return output["choices"][0]["text"].strip()
            else:
                return "無法生成回答"
        except Exception as e:
            print(f"生成回答時出錯: {str(e)}")
            return f"生成過程中出錯: {str(e)}"
import os
import logging
from litellm import completion, embedding
from dotenv import load_dotenv

# 載入 .env 環境變數
load_dotenv()

logger = logging.getLogger(__name__)

# LiteLLM 自己掛了 handler 又往 root 傳,每次呼叫都會印兩次 INFO。只留警告以上
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def embed_model() -> str:
    """目前使用的嵌入模型,由環境變數 RAG_EMBED_MODEL 決定。

    建立向量庫與查詢都必須用同一個模型,所以兩邊都從這裡取。
    """
    return os.environ.get("RAG_EMBED_MODEL", "ollama/nomic-embed-text")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """批次取得多段文本的向量。"""
    model = embed_model()
    try:
        response = embedding(model=model, input=texts)
        return [item["embedding"] for item in response.data]
    except Exception as e:
        logger.error(f"Embedding error with model {model}: {e}")
        raise


def get_embedding(text: str) -> list[float]:
    """取得單一文本的向量。"""
    return get_embeddings([text])[0]


def get_llm_response(sys_prompt: str, user_prompt: str) -> str:
    """
    獲取 LLM 回應，並強制輸出為 JSON 格式。
    模型由環境變數 RAG_LLM_MODEL 決定。
    """
    model = os.environ.get("RAG_LLM_MODEL", "ollama/llama3.1:8b")

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # litellm 會自動處理各平台的 JSON 模式轉換
        response = completion(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM completion error with model {model}: {e}")
        raise

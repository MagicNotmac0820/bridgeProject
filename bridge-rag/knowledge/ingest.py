"""知識庫建置:PDF → 切塊(parsed/*.jsonl)→ 嵌入並寫入 ChromaDB。

在 bridge-rag 目錄下執行:
    python -m knowledge.ingest                 切塊檔不存在才解析 PDF,再建 v1 的向量庫
    python -m knowledge.ingest --reparse       強制重新解析 PDF(會覆蓋 parsed/*.jsonl)

parsed/*.jsonl 已進版控,被某個版本用過就不要覆蓋,否則舊版本重現不了。
"""

import argparse
import os
import json
import logging
import re
import pymupdf
import tiktoken
import chromadb

from versions.llm import embed_model, get_embeddings

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_BASE_DIR, "chroma_db")
PARSED_DIR = os.path.join(_BASE_DIR, "parsed")


def collection_name(version: str, model: str) -> str:
    """每個版本、每個嵌入模型各用一個 collection,互不覆蓋。

    例如 ("v1", "ollama/nomic-embed-text") → "v1-ollama-nomic-embed-text"
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{version}-{model}").strip("-._")

def get_tokenizer():
    """取得 tiktoken tokenizer，使用 cl100k_base encoding"""
    return tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    """計算文本的 token 數量"""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def chunk_text(text: str, chunk_size: int, overlap: int, base_metadata: dict) -> list[dict]:
    """
    將文本分塊，並附加 metadata
    
    Args:
        text (str): 要分塊的文本
        chunk_size (int): 每個 chunk 的最大 token 數
        overlap (int): chunk 之間的重疊 token 數
        base_metadata (dict): 基礎的 metadata
        
    Returns:
        list[dict]: 包含 doc_id, text, metadata 的字典列表
    """
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = tokenizer.decode(chunk_tokens)
        
        metadata = base_metadata.copy()
        metadata["chunk_index"] = chunk_index
        # 確保 doc_id 唯一
        doc_id = f"{metadata['doc_id']}_{chunk_index}"
        
        chunks.append({
            "doc_id": doc_id,
            "text": chunk_text_str,
            "metadata": metadata
        })
        
        if end == len(tokens):
            break
            
        start += (chunk_size - overlap)
        chunk_index += 1
        
    return chunks

def parse_laws_pdf(pdf_path: str) -> list[dict]:
    """
    解析橋牌法規 PDF
    以 Law XX 為邊界進行分塊
    
    Args:
        pdf_path (str): PDF 檔案路徑
        
    Returns:
        list[dict]: 解析後的 chunks
    """
    doc = pymupdf.open(pdf_path)
    chunks = []
    current_law = "Unknown"
    current_text = ""
    current_page = 0
    
    # 簡單的正則表示式，尋找類似 "Law 1", "LAW 25" 的標題
    law_pattern = re.compile(r"^(LAW|Law)\s+(\d+[A-Z]?)", re.IGNORECASE)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = law_pattern.match(line)
            if match:
                # 儲存前一個 law 的區塊
                if current_text:
                    chunks.extend(chunk_text(current_text, 500, 50, {
                        "doc_id": f"law_2017_{current_law.lower().replace(' ', '_')}",
                        "source": os.path.basename(pdf_path),
                        "type": "rule",
                        "chapter": "General", # 簡化處理
                        "law_number": current_law,
                        "page": current_page
                    }))
                current_law = f"Law {match.group(2)}"
                current_text = line + "\n"
                current_page = page_num + 1
            else:
                current_text += line + "\n"
                
    # 處理最後一個 block
    if current_text:
        chunks.extend(chunk_text(current_text, 500, 50, {
            "doc_id": f"law_2017_{current_law.lower().replace(' ', '_')}",
            "source": os.path.basename(pdf_path),
            "type": "rule",
            "chapter": "General",
            "law_number": current_law,
            "page": current_page
        }))
        
    doc.close()
    return chunks

def parse_sp3_pdf(pdf_path: str) -> list[dict]:
    """
    解析 SP3 叫牌制度卡 PDF
    按叫牌主題區塊分塊
    
    Args:
        pdf_path (str): PDF 檔案路徑
        
    Returns:
        list[dict]: 解析後的 chunks
    """
    doc = pymupdf.open(pdf_path)
    chunks = []
    
    current_context = "General"
    context_keywords = ["OPENING", "RESPONSES", "OVERCALLS", "COMPETITIVE", "DEFENSE", "LEADS"]
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        lines = text.split('\n')
        
        current_text = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 檢查是否為新主題
            upper_line = line.upper()
            is_new_context = False
            for kw in context_keywords:
                if kw in upper_line and len(line) < 30: # 假設標題不會太長
                    if current_text:
                        chunks.extend(chunk_text(current_text, 300, 100, {
                            "doc_id": f"sp3_{current_context.lower().replace(' ', '_')}_{page_num+1}",
                            "source": "SP3",
                            "type": "system_card",
                            "bidding_context": current_context,
                            "category": current_context,
                            "page": page_num + 1
                        }))
                        current_text = ""
                    current_context = line
                    is_new_context = True
                    break
                    
            current_text += line + "\n"
            
        if current_text:
            chunks.extend(chunk_text(current_text, 300, 100, {
                "doc_id": f"sp3_{current_context.lower().replace(' ', '_')}_{page_num+1}_end",
                "source": "SP3",
                "type": "system_card",
                "bidding_context": current_context,
                "category": current_context,
                "page": page_num + 1
            }))
            
    doc.close()
    return chunks

def write_to_jsonl(chunks: list, output_path: str):
    """
    將分塊結果寫入 JSONL 檔案
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    logger.info(f"已寫入 {len(chunks)} 筆資料至 {output_path}")

def load_parsed_chunks(parsed_dir: str = PARSED_DIR) -> list[dict]:
    """讀取 parsed 目錄下所有 JSONL,重複的 doc_id 加上流水號。依檔名排序,確保每次順序一致。"""
    chunks = []
    seen_ids = set()
    for filename in sorted(os.listdir(parsed_dir)):
        if not filename.endswith(".jsonl"):
            continue
        with open(os.path.join(parsed_dir, filename), 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                base_id = data["doc_id"]
                doc_id = base_id
                counter = 1
                while doc_id in seen_ids:
                    doc_id = f"{base_id}_{counter}"
                    counter += 1
                seen_ids.add(doc_id)
                chunks.append({"doc_id": doc_id, "text": data["text"], "metadata": data["metadata"]})
    return chunks


def ingest_to_chroma(name: str, chroma_path: str = CHROMA_PATH, parsed_dir: str = PARSED_DIR):
    """
    將 parsed 目錄下的 JSONL 資料嵌入並寫入 ChromaDB 的 `name` collection。

    嵌入模型取自 versions.llm.embed_model(),並記在 collection 的 metadata
    (embed_model),讓讀取端可以確認查詢與建庫用的是同一個模型。
    collection 已有資料就跳過,不會覆蓋。
    """
    if not os.path.exists(parsed_dir):
        logger.warning(f"Parsed directory {parsed_dir} does not exist. Nothing to ingest.")
        return

    model = embed_model()
    client = chromadb.PersistentClient(path=chroma_path)

    # 向量一律自己算好再寫入,不讓 ChromaDB 用它內建的嵌入模型
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "embed_model": model},
    )

    if collection.count() > 0:
        logger.info(f"ChromaDB collection '{name}' 已有 {collection.count()} 筆資料，跳過寫入。")
        return

    chunks = load_parsed_chunks(parsed_dir)
    if not chunks:
        logger.info("沒有資料可以寫入 ChromaDB。")
        return

    logger.info(f"使用 {model} 進行嵌入,寫入 collection '{name}'")
    # 批次寫入，避免一次寫入太多導致記憶體問題或 API 限制
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        collection.add(
            ids=[c["doc_id"] for c in batch],
            documents=texts,
            metadatas=[c["metadata"] for c in batch],
            embeddings=get_embeddings(texts),
        )
        logger.info(f"ChromaDB 寫入進度: {i + len(batch)}/{len(chunks)}")

    logger.info(f"成功將 {len(chunks)} 筆資料寫入 ChromaDB。")


def parse_pdfs(raw_dir: str, parsed_dir: str):
    """解析 raw 目錄下的 PDF,寫成 parsed/*.jsonl。"""
    os.makedirs(parsed_dir, exist_ok=True)
    sources = [
        ("2017LawsofDuplicateBridge-paginated.pdf", parse_laws_pdf, "laws.jsonl"),
        ("SP3 (bk) single pages.pdf", parse_sp3_pdf, "sp3.jsonl"),
    ]
    for pdf_name, parser, out_name in sources:
        pdf_path = os.path.join(raw_dir, pdf_name)
        if os.path.exists(pdf_path):
            logger.info(f"開始解析 PDF: {pdf_path}")
            write_to_jsonl(parser(pdf_path), os.path.join(parsed_dir, out_name))
        else:
            logger.warning(f"找不到檔案: {pdf_path}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v1", help="要建哪個版本的 collection,預設 v1")
    p.add_argument("--reparse", action="store_true", help="重新解析 PDF,覆蓋 parsed/*.jsonl")
    args = p.parse_args()

    has_parsed = os.path.isdir(PARSED_DIR) and any(f.endswith(".jsonl") for f in os.listdir(PARSED_DIR))
    if args.reparse or not has_parsed:
        parse_pdfs(os.path.join(_BASE_DIR, "raw"), PARSED_DIR)
    else:
        logger.info(f"{PARSED_DIR} 已有切塊資料,略過 PDF 解析(要重新解析請加 --reparse)")

    ingest_to_chroma(collection_name(args.version, embed_model()))


if __name__ == "__main__":
    main()

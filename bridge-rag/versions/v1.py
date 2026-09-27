import os
import json
import logging
import traceback
import chromadb
from rank_bm25 import BM25Okapi

from knowledge.ingest import CHROMA_PATH, collection_name, ingest_to_chroma
from versions.common import describe_hand, format_auction, format_contract, format_current_trick
from versions.llm import embed_model, get_embedding, get_llm_response

logger = logging.getLogger(__name__)

DESCRIPTION = "RAG v1 with Hybrid Retrieval (ChromaDB + BM25) and RRF, using Model-Agnostic LiteLLM"

# 全域變數
_chroma_client = None
_collection = None
_bm25 = None
_bm25_docs = [] # 儲存文件對應的 metadata 和原文: [{"id": doc_id, "text": text, "metadata": metadata}]

_top_k = int(os.environ.get('RAG_TOP_K', '5'))

def _open_collection(client, name: str, model: str):
    """取得 collection,不存在或是空的就建立。建庫用的模型和現在不同就直接失敗。"""
    try:
        collection = client.get_collection(name=name)
    except Exception:
        collection = None

    if collection is None or collection.count() == 0:
        logger.info(f"Collection '{name}' is empty or missing. Running ingest...")
        ingest_to_chroma(name, chroma_path=CHROMA_PATH)
        collection = client.get_collection(name=name)

    # 查詢向量和庫裡的向量出自不同模型時不會報錯,只會檢索到錯的東西,所以啟動時就擋下
    built_with = (collection.metadata or {}).get("embed_model")
    if built_with != model:
        raise RuntimeError(
            f"collection '{name}' 是用 {built_with!r} 建的,目前 RAG_EMBED_MODEL={model!r}。"
            f"請刪除 knowledge/chroma_db 後重新啟動以重建")
    return collection


def setup():
    """
    初始化 RAG 服務：
    1. 連接 ChromaDB,取得 v1 專屬的 collection(名稱含嵌入模型)
    2. 如果為空則填充資料
    3. 建立 BM25 索引
    """
    global _chroma_client, _collection, _bm25, _bm25_docs

    model = embed_model()
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    _collection = _open_collection(_chroma_client, collection_name("v1", model), model)

    # 建立 BM25 索引
    logger.info("Building BM25 index from ChromaDB documents...")
    all_data = _collection.get()
    
    tokenized_corpus = []
    _bm25_docs = []
    
    if all_data and all_data.get('ids'):
        for i in range(len(all_data['ids'])):
            doc_id = all_data['ids'][i]
            text = all_data['documents'][i]
            metadata = all_data['metadatas'][i] if all_data['metadatas'] else {}
            
            _bm25_docs.append({"id": doc_id, "text": text, "metadata": metadata})
            # 簡單的空白斷詞供 BM25 使用
            tokenized_corpus.append(text.lower().split())
            
    if tokenized_corpus:
        _bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(tokenized_corpus)} documents.")


def _hybrid_retrieve(query: str, request_type: str, top_k: int = 5) -> list[dict]:
    """
    混合檢索：
    1. ChromaDB 向量檢索 (top_k * 2)
    2. BM25 關鍵字檢索 (top_k * 2)
    3. 透過 Reciprocal Rank Fusion (RRF) 合併排序
    """
    if not _collection or not _bm25:
        return []

    # 針對叫牌進行 Metadata 過濾
    where_filter = {"type": "system_card"} if request_type == "bid" else None
    
    # === 1. Vector Search (ChromaDB) ===
    try:
        query_embedding = get_embedding(query)
        
        chroma_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k * 2
        }
        if where_filter:
            chroma_kwargs["where"] = where_filter
            
        chroma_res = _collection.query(**chroma_kwargs)
    except Exception as e:
        logger.error(f"ChromaDB retrieval error: {e}")
        chroma_res = {"ids": [[]], "documents": [[]], "metadatas": [[]]}

    vector_results = []
    if chroma_res.get("ids") and chroma_res["ids"][0]:
        for i in range(len(chroma_res["ids"][0])):
            vector_results.append({
                "id": chroma_res["ids"][0][i],
                "text": chroma_res["documents"][0][i]
            })

    # === 2. Keyword Search (BM25) ===
    tokenized_query = query.lower().split()
    bm25_scores = _bm25.get_scores(tokenized_query)
    
    bm25_scored_docs = []
    for i, score in enumerate(bm25_scores):
        doc = _bm25_docs[i]
        # 套用 Metadata 過濾條件
        if where_filter and doc["metadata"].get("type") != where_filter["type"]:
            continue
        bm25_scored_docs.append((score, doc))
        
    bm25_scored_docs.sort(key=lambda x: x[0], reverse=True)
    bm25_results = [doc for score, doc in bm25_scored_docs[:top_k * 2]]

    # === 3. RRF (Reciprocal Rank Fusion) 合併 ===
    rrf_k = 60
    rrf_scores = {}
    
    for rank, doc in enumerate(vector_results):
        doc_id = doc["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"score": 0.0, "doc": doc, "sources": []}
        rrf_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)
        rrf_scores[doc_id]["sources"].append("vector")
        
    for rank, doc in enumerate(bm25_results):
        doc_id = doc["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"score": 0.0, "doc": doc, "sources": []}
        rrf_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)
        rrf_scores[doc_id]["sources"].append("bm25")

    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    
    # === 4. 回傳格式化結果 ===
    final_results = []
    for item in sorted_results[:top_k]:
        final_results.append({
            "doc_id": item["doc"]["id"],
            "snippet": item["doc"]["text"],
            "score": item["score"],
            "source": "+".join(item["sources"])
        })
        
    return final_results


def _get_fallback_action(legal_actions, default_action=None):
    """
    提供 Fallback 的合法動作
    """
    if default_action and default_action in legal_actions:
        return default_action
    return legal_actions[0] if legal_actions else ""


def choose_bid(request):
    """
    依據 request 回傳 (叫品, 解釋, 檢索結果) 三元組
    """
    try:
        # 1. 描述手牌與叫牌狀態
        hand_desc = describe_hand(request.hand)
        fmt_auction = format_auction(request.auction, request.dealer)
        
        # 2. 構造查詢字串
        query = f"Bidding with {hand_desc} after {fmt_auction}"
        
        # 3. 混合檢索
        retrieved = _hybrid_retrieve(query, request_type="bid", top_k=_top_k)
        retrieved_context = "\n\n".join([r["snippet"] for r in retrieved])
        
        # 4. 組裝 prompt 並呼叫 LLM
        sys_prompt = (
            "You are an expert bridge player following the SP3 bidding system strictly.\n"
            "You MUST choose EXACTLY ONE bid from the Legal Bids list. No other bid is acceptable.\n"
            "Output a JSON object with \"thought\" (your reasoning) and \"action\" (the chosen bid)."
        )
        
        user_prompt = (
            f"## Your Hand\n{hand_desc}\n\n"
            f"## Auction So Far\n{fmt_auction}\n\n"
            f"## Vulnerability\n{request.vulnerability}\n\n"
            f"## Relevant System Guidelines\n{retrieved_context}\n\n"
            f"## Legal Bids\n{request.legal_bids}\n\n"
            f"Respond with JSON ONLY: {{\"thought\": \"...\", \"action\": \"...\"}}"
        )
        
        # 5. 解析回應與驗證
        content = get_llm_response(sys_prompt, user_prompt)
        result = json.loads(content)
        
        action = result.get("action")
        thought = result.get("thought", "")
        
        if action not in request.legal_bids:
            raise ValueError(f"LLM returned illegal action: {action}")
            
        return action, thought, retrieved

    except Exception as e:
        logger.error(f"Error in choose_bid: {e}\n{traceback.format_exc()}")
        fallback = _get_fallback_action(request.legal_bids, default_action='P')
        return fallback, f"Fallback due to error: {e}", []


def choose_card(request):
    """
    依據 request 回傳 (牌張, 解釋, 檢索結果) 三元組
    """
    try:
        # 1. 描述手牌、合約與當前墩狀態
        hand_desc = f"Hand: {', '.join(request.hand)}"
        if request.dummy:
            hand_desc += f"\nDummy: {', '.join(request.dummy)}"
            
        fmt_contract = format_contract(request.contract)
        fmt_trick = format_current_trick(request.current_trick)
        
        # 2. 構造查詢字串
        query = f"Playing {fmt_contract}, trick: {fmt_trick}, hand: {hand_desc}"
        
        # 3. 混合檢索 (不過濾 type)
        retrieved = _hybrid_retrieve(query, request_type="play", top_k=_top_k)
        retrieved_context = "\n\n".join([r["snippet"] for r in retrieved])
        
        # 4. 組裝 prompt 並呼叫 LLM
        sys_prompt = (
            "You are an expert bridge player.\n"
            "You MUST choose EXACTLY ONE card from the Legal Cards list. No other card is acceptable.\n"
            "Output a JSON object with \"thought\" (your reasoning) and \"action\" (the chosen card)."
        )
        
        user_prompt = (
            f"## Context\nContract: {fmt_contract}\nPlaying for: {request.playing_for}\n\n"
            f"## Hands\n{hand_desc}\n\n"
            f"## Current Trick\n{fmt_trick}\n\n"
            f"## Relevant Tips\n{retrieved_context}\n\n"
            f"## Legal Cards\n{request.legal_cards}\n\n"
            f"Respond with JSON ONLY: {{\"thought\": \"...\", \"action\": \"...\"}}"
        )
        
        # 5. 解析回應與驗證
        content = get_llm_response(sys_prompt, user_prompt)
        result = json.loads(content)
        
        action = result.get("action")
        thought = result.get("thought", "")
        
        if action not in request.legal_cards:
            raise ValueError(f"LLM returned illegal action: {action}")
            
        return action, thought, retrieved

    except Exception as e:
        logger.error(f"Error in choose_card: {e}\n{traceback.format_exc()}")
        fallback = _get_fallback_action(request.legal_cards)
        return fallback, f"Fallback due to error: {e}", []

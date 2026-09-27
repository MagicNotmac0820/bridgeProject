"""v1:向量庫建置、模型一致性檢查、混合檢索與決策流程。

嵌入與 LLM 都換成假的,不需要 Ollama 或任何 API 金鑰。
"""

import hashlib
import json

import chromadb
import pytest

from knowledge import ingest
from rag_service import BidRequest, PlayRequest
from test_service import requests_from_real_deal
from versions import v1

MODEL = "fake/embed"

DOCS = [
    {"doc_id": "sp3_opening_1", "text": "Opening bids: 1NT shows 15-17 HCP balanced",
     "metadata": {"type": "system_card", "page": 1}},
    {"doc_id": "sp3_responses_1", "text": "Responses to 1 of a major: raise with three card support",
     "metadata": {"type": "system_card", "page": 2}},
    {"doc_id": "law_2017_law_44", "text": "Law 44 sequence and procedure of play, follow suit",
     "metadata": {"type": "rule", "page": 50}},
]


def fake_embeddings(texts):
    return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


@pytest.fixture
def kb(tmp_path, monkeypatch):
    """在暫存目錄準備切塊資料與向量庫路徑,嵌入換成假的。"""
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "docs.jsonl").write_text("\n".join(json.dumps(d) for d in DOCS), encoding="utf-8")
    chroma = str(tmp_path / "chroma_db")

    monkeypatch.setenv("RAG_EMBED_MODEL", MODEL)
    monkeypatch.setattr(ingest, "get_embeddings", fake_embeddings)
    monkeypatch.setattr(v1, "get_embedding", lambda t: fake_embeddings([t])[0])
    monkeypatch.setattr(v1, "CHROMA_PATH", chroma)
    real_ingest = ingest.ingest_to_chroma
    monkeypatch.setattr(v1, "ingest_to_chroma",
                        lambda name, chroma_path: real_ingest(name, chroma_path=chroma, parsed_dir=str(parsed)))
    return chroma


def test_collection_name_is_per_version_and_model():
    assert ingest.collection_name("v1", "ollama/nomic-embed-text") == "v1-ollama-nomic-embed-text"
    assert ingest.collection_name("v1", "a") != ingest.collection_name("v2", "a")


def test_setup_builds_collection_and_records_model(kb):
    v1.setup()
    assert v1._collection.name == "v1-fake-embed"
    assert v1._collection.count() == len(DOCS)
    assert v1._collection.metadata["embed_model"] == MODEL
    assert len(v1._bm25_docs) == len(DOCS)


def test_model_mismatch_fails_loudly(kb):
    v1.setup()
    client = chromadb.PersistentClient(path=kb)
    with pytest.raises(RuntimeError, match="fake/embed"):
        v1._open_collection(client, "v1-fake-embed", "other/model")


def test_bid_retrieval_only_uses_system_card(kb):
    v1.setup()
    results = v1._hybrid_retrieve("opening 1NT balanced", request_type="bid", top_k=5)
    assert results and all(r["doc_id"].startswith("sp3_") for r in results)
    assert {"doc_id", "snippet", "score", "source"} <= set(results[0])


def test_choose_bid_returns_llm_bid_with_retrieved(kb, monkeypatch):
    v1.setup()
    bid_req, _ = requests_from_real_deal()
    choice = bid_req["legal_bids"][1]
    monkeypatch.setattr(v1, "get_llm_response",
                        lambda s, u: json.dumps({"thought": "because", "action": choice}))
    bid, explanation, retrieved = v1.choose_bid(BidRequest(**bid_req))
    assert (bid, explanation) == (choice, "because")
    assert retrieved


def test_choose_bid_illegal_answer_falls_back_to_pass(kb, monkeypatch):
    v1.setup()
    bid_req, _ = requests_from_real_deal()
    monkeypatch.setattr(v1, "get_llm_response", lambda s, u: json.dumps({"thought": "", "action": "8S"}))
    bid, explanation, retrieved = v1.choose_bid(BidRequest(**bid_req))
    assert bid == "P" and explanation.startswith("Fallback due to error") and retrieved == []


def test_choose_card_returns_legal_card(kb, monkeypatch):
    v1.setup()
    _, play_req = requests_from_real_deal()
    choice = play_req["legal_cards"][-1]
    monkeypatch.setattr(v1, "get_llm_response",
                        lambda s, u: json.dumps({"thought": "t", "action": choice}))
    card, _, retrieved = v1.choose_card(PlayRequest(**play_req))
    assert card == choice and retrieved

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from zhipu_client import ZhipuClient
from memory_system import AdvancedJsonCardStore, ConversationMemory, VectorMemoryIndex
from rag_agent import RAGAgent, ZhipuEmbedder


DATA_DIR = "memory_data"
MEMORY_PATH = os.path.join(DATA_DIR, "conversation.json")
VEC_INDEX_PATH = os.path.join(DATA_DIR, "vector_index.json")
EMBED_CACHE_PATH = os.path.join(DATA_DIR, "embed_cache.json")
CARDS_PATH = os.path.join(DATA_DIR, "cards.json")


def _vec_chunk_size() -> int:
    # 允许通过环境变量调小 chunk，降低单条 embedding input 的长度，避免请求过大
    try:
        return max(1, int(os.getenv("VEC_CHUNK_SIZE", "20")))
    except ValueError:
        return 20


def _vec_index_needs_rebuild(vec_index: VectorMemoryIndex, memory: ConversationMemory, chunk_size: int) -> bool:
    if not memory.turns:
        return False
    expected_chunks = (len(memory.turns) + chunk_size - 1) // chunk_size
    if len(vec_index.chunks) != expected_chunks:
        return True
    # 代码升级后：旧 vector_index.json 可能没有 meta 或使用旧格式；强制重建一次
    if not getattr(vec_index, "meta", None):
        return True
    if (vec_index.meta or {}).get("embed_doc_format") != "compact_v1":
        return True
    return False


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


class ChatRequest(BaseModel):
    message: str
    mode: str = "rag"  # "rag" | "chat"
    top_k: int = 4


class ChatResponse(BaseModel):
    answer: str
    mode: str
    retrieved: Optional[List[Dict[str, Any]]] = None


class CardUpsertRequest(BaseModel):
    card_id: str
    facts: Dict[str, Any]


app = FastAPI(title="Memory Chat (ZhipuAI + RAG)")

@app.exception_handler(Exception)
async def _unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    # 确保前端永远收到 JSON（避免 resp.json() 因为纯文本/HTML 而崩溃）
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器异常：{exc.__class__.__name__}: {exc}"},
    )


@app.on_event("startup")
def _startup() -> None:
    _ensure_data_dir()


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


def _load_state():
    memory = ConversationMemory.load(MEMORY_PATH)
    cards = AdvancedJsonCardStore.load(CARDS_PATH)
    vec_index = VectorMemoryIndex.load(VEC_INDEX_PATH)
    return memory, cards, vec_index


def _save_state(memory: ConversationMemory, cards: AdvancedJsonCardStore, vec_index: VectorMemoryIndex) -> None:
    memory.save(MEMORY_PATH)
    cards.save(CARDS_PATH)
    vec_index.save(VEC_INDEX_PATH)


def _zhipu_client() -> ZhipuClient:
    try:
        return ZhipuClient()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"ZhipuClient 初始化失败：{e}。请设置环境变量 ZHIPU_API_KEY（必要时设置模型名）。",
        )


def _maybe_store_explicit_note(cards: AdvancedJsonCardStore, user_text: str) -> None:
    """
    用户明确要求“记住/记一下”的内容，写入 JSON Cards，确保长期可回忆。
    注意：我们只在用户显式触发时写入，避免把所有对话都塞进常驻 prompt。
    """
    t = (user_text or "").strip()
    if not t:
        return
    if not any(k in t for k in ("记住", "记一下", "帮我记", "请记", "记下")):
        return

    # 过滤掉纯指令句，尽量提取“要记的内容”
    # e.g. "帮我记一下就行" -> 不写；"你记一下：xxx" -> 写入 xxx
    payload = t
    for sep in ("：", ":", "—", "-", "，", ","):
        if sep in t:
            left, right = t.split(sep, 1)
            if any(k in left for k in ("记住", "记一下", "帮我记", "请记", "记下")) and right.strip():
                payload = right.strip()
                break
    if any(payload == x for x in ("帮我记一下就行", "你记一下", "记一下", "记住", "请记住", "帮我记")):
        return

    # 追加到 notes card
    notes_card_id = "notes"
    existing = cards.cards.get(notes_card_id)
    notes = []
    if existing and isinstance(existing.facts, dict):
        notes = list(existing.facts.get("notes", []) or [])
    notes.append({"text": payload, "created_at": __import__("datetime").datetime.now().isoformat(timespec="seconds")})
    cards.upsert(notes_card_id, {"notes": notes})


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message 不能为空")
    if req.mode not in ("rag", "chat"):
        raise HTTPException(status_code=400, detail="mode 必须是 rag 或 chat")

    memory, cards, vec_index = _load_state()
    client = _zhipu_client()

    # 记录用户消息
    memory.add_turn("用户", msg)
    _maybe_store_explicit_note(cards, msg)

    retrieved = None
    if req.mode == "chat":
        answer = client.chat_completions(
            messages=[
                {"role": "system", "content": "你是一个中文助理，请直接回答用户问题。"},
                {"role": "user", "content": msg},
            ]
        )
    else:
        # 确保向量索引可用：仅在需要时重建（避免每轮都 build，成本高也更容易触发上游限制）
        embedder = ZhipuEmbedder(client)
        cs = _vec_chunk_size()
        if _vec_index_needs_rebuild(vec_index, memory, cs):
            vec_index.build(memory, embedder=embedder, chunk_size=cs, cache_path=EMBED_CACHE_PATH)

        # 先检索，给前端展示
        retrieved = vec_index.search(msg, embedder=embedder, top_k=max(1, int(req.top_k)))

        agent = RAGAgent(client=client, vector_index=vec_index, cards=cards, embed_cache_path=EMBED_CACHE_PATH)
        answer = agent.ask(msg, top_k=max(1, int(req.top_k)))

    # 记录助手消息
    memory.add_turn("助手", answer)

    # 不在每轮结尾强制 build；下次请求时按需增量重建即可

    _save_state(memory, cards, vec_index)
    return ChatResponse(answer=answer, mode=req.mode, retrieved=retrieved)


@app.get("/api/cards")
def list_cards() -> Dict[str, Any]:
    _ensure_data_dir()
    cards = AdvancedJsonCardStore.load(CARDS_PATH)
    return cards.to_dict()


@app.post("/api/cards")
def upsert_card(req: CardUpsertRequest) -> Dict[str, Any]:
    _ensure_data_dir()
    cards = AdvancedJsonCardStore.load(CARDS_PATH)
    cards.upsert(req.card_id, req.facts)
    cards.save(CARDS_PATH)
    return {"ok": True}


@app.post("/api/search")
def search_memory(query: Dict[str, Any]) -> Dict[str, Any]:
    q = str(query.get("query", "")).strip()
    top_k = int(query.get("top_k", 4))
    if not q:
        raise HTTPException(status_code=400, detail="query 不能为空")

    memory, _cards, vec_index = _load_state()
    client = _zhipu_client()
    embedder = ZhipuEmbedder(client)
    if not vec_index.chunks and memory.turns:
        vec_index.build(memory, embedder=embedder, chunk_size=_vec_chunk_size(), cache_path=EMBED_CACHE_PATH)
        vec_index.save(VEC_INDEX_PATH)

    results = vec_index.search(q, embedder=embedder, top_k=max(1, top_k))
    return {"results": results}


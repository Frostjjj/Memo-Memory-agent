import json
import os
from typing import Optional

from memory_system import (
    AdvancedJsonCardStore,
    ConversationMemory,
    LocalHashEmbedder,
    MemoryIndex,
    VectorMemoryIndex,
    search_user_memory,
)
from zhipu_client import ZhipuClient
from rag_agent import RAGAgent, ZhipuEmbedder


DATA_DIR = "memory_data"
MEMORY_PATH = os.path.join(DATA_DIR, "conversation.json")
INDEX_PATH = os.path.join(DATA_DIR, "index.json")
VEC_INDEX_PATH = os.path.join(DATA_DIR, "vector_index.json")
EMBED_CACHE_PATH = os.path.join(DATA_DIR, "embed_cache.json")
CARDS_PATH = os.path.join(DATA_DIR, "cards.json")


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _parse_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _help() -> str:
    return (
        "命令：\n"
        "  /help                显示帮助\n"
        "  /user <内容>          追加用户发言\n"
        "  /assistant <内容>     追加助手发言\n"
        "  /search <查询>        检索对话记忆\n"
        "  /reindex [窗口大小]   重建索引（默认20）\n"
        "  /reindex_vec [窗口大小]  重建向量索引（默认20，需要 ZHIPU_API_KEY）\n"
        "  /search_vec <查询>    向量检索对话记忆（需要 ZHIPU_API_KEY）\n"
        "  /ask <问题>           RAG 问答（向量检索 + JSON Cards + 智谱生成）\n"
        "  /card <json>          写入 Advanced JSON Card\n"
        "  /facts               查看所有 JSON Cards\n"
        "  /exit                保存并退出\n"
        "\n"
        "示例：\n"
        "  /card {\"card_id\":\"travel\",\"facts\":{\"护照过期\":\"2025-02\",\"已预订\":\"1月15日飞东京\"}}\n"
        "  /search 最终转账指令\n"
    )


def main() -> None:
    _ensure_data_dir()
    memory = ConversationMemory.load(MEMORY_PATH)
    cards = AdvancedJsonCardStore.load(CARDS_PATH)
    index = MemoryIndex.load(INDEX_PATH)
    vec_index = VectorMemoryIndex.load(VEC_INDEX_PATH)
    if not index.chunks and memory.turns:
        index.build(memory, chunk_size=20)

    print("交互式记忆系统已启动，输入 /help 查看命令。")
    while True:
        raw = input(">>> ").strip()
        if not raw:
            continue

        if raw == "/help":
            print(_help())
            continue

        if raw == "/exit":
            memory.save(MEMORY_PATH)
            index.save(INDEX_PATH)
            cards.save(CARDS_PATH)
            print("已保存并退出。")
            break

        if raw.startswith("/user "):
            memory.add_turn("用户", raw[len("/user ") :])
            print("已记录用户发言。")
            continue

        if raw.startswith("/assistant "):
            memory.add_turn("助手", raw[len("/assistant ") :])
            print("已记录助手发言。")
            continue

        if raw.startswith("/reindex"):
            parts = raw.split()
            chunk_size = 20
            if len(parts) > 1 and parts[1].isdigit():
                chunk_size = int(parts[1])
            index.build(memory, chunk_size=chunk_size)
            index.save(INDEX_PATH)
            print(f"索引已更新，窗口大小 {chunk_size}。")
            continue

        if raw.startswith("/reindex_vec"):
            parts = raw.split()
            chunk_size = 20
            if len(parts) > 1 and parts[1].isdigit():
                chunk_size = int(parts[1])
            try:
                client = ZhipuClient()
            except Exception as e:
                print(f"初始化 ZhipuClient 失败：{e}")
                print("请先设置环境变量 ZHIPU_API_KEY（以及必要时的模型名）。")
                continue
            embedder = ZhipuEmbedder(client)
            vec_index.build(memory, embedder=embedder, chunk_size=chunk_size, cache_path=EMBED_CACHE_PATH)
            vec_index.save(VEC_INDEX_PATH)
            print(f"向量索引已更新，窗口大小 {chunk_size}。")
            continue

        if raw.startswith("/search_vec "):
            query = raw[len("/search_vec ") :]
            if not vec_index.chunks:
                print("向量索引为空，请先 /reindex_vec。")
                continue
            try:
                client = ZhipuClient()
            except Exception as e:
                print(f"初始化 ZhipuClient 失败：{e}")
                print("请先设置环境变量 ZHIPU_API_KEY（以及必要时的模型名）。")
                continue
            embedder = ZhipuEmbedder(client)
            results = vec_index.search(query, embedder=embedder, top_k=3)
            if not results:
                print("未检索到相关记忆。")
                continue
            for i, r in enumerate(results, 1):
                print(f"[{i}] {r['prefix_summary']} (score={r['score']})")
                print(r["text"])
                print("-" * 40)
            continue

        if raw.startswith("/ask "):
            question = raw[len("/ask ") :]
            if not vec_index.chunks:
                print("向量索引为空，请先 /reindex_vec。")
                continue
            try:
                client = ZhipuClient()
            except Exception as e:
                print(f"初始化 ZhipuClient 失败：{e}")
                print("请先设置环境变量 ZHIPU_API_KEY（以及必要时的模型名）。")
                continue
            agent = RAGAgent(client=client, vector_index=vec_index, cards=cards, embed_cache_path=EMBED_CACHE_PATH)
            answer = agent.ask(question, top_k=4)
            print(answer)
            memory.add_turn("助手", answer)
            continue

        if raw.startswith("/search "):
            query = raw[len("/search ") :]
            if not index.chunks:
                index.build(memory, chunk_size=20)
            results = search_user_memory(query, index, top_k=3)
            if not results:
                print("未检索到相关记忆。")
                continue
            for i, r in enumerate(results, 1):
                print(f"[{i}] {r['prefix_summary']} (score={r['score']})")
                print(r["text"])
                print("-" * 40)
            continue

        if raw.startswith("/card "):
            payload = _parse_json(raw[len("/card ") :])
            if not payload or "card_id" not in payload or "facts" not in payload:
                print("JSON 格式错误或缺少 card_id/facts。")
                continue
            cards.upsert(payload["card_id"], payload["facts"])
            cards.save(CARDS_PATH)
            print("JSON Card 已保存。")
            continue

        if raw == "/facts":
            all_cards = cards.list_cards()
            if not all_cards:
                print("暂无 JSON Cards。")
                continue
            for c in all_cards:
                print(f"- {c.card_id} @ {c.updated_at}")
                print(json.dumps(c.facts, ensure_ascii=False))
            continue

        memory.add_turn("用户", raw)
        print("已记录。可用 /assistant 追加助手回合。")


if __name__ == "__main__":
    main()

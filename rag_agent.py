import json
import os
from typing import Dict, List, Optional

from zhipu_client import ZhipuClient
from memory_system import AdvancedJsonCardStore, Embedder, VectorMemoryIndex


class ZhipuEmbedder(Embedder):
    def __init__(self, client: ZhipuClient):
        self.client = client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.client.embeddings(texts)


def _format_cards(cards: AdvancedJsonCardStore) -> str:
    lines = []
    for c in cards.list_cards():
        lines.append(f"- {c.card_id} @ {c.updated_at}: {json.dumps(c.facts, ensure_ascii=False)}")
    return "\n".join(lines) if lines else "(无)"


def _format_memories(memories: List[Dict]) -> str:
    if not memories:
        return "(无)"
    parts = []
    for m in memories:
        txt = m.get("text_compact") or m.get("text") or ""
        parts.append(f"{m['prefix_summary']} (score={m['score']})\n{txt}")
    return "\n\n---\n\n".join(parts)


class RAGAgent:
    """
    双层记忆（JSON Cards + 向量检索对话块） + 智谱大模型生成。
    """

    def __init__(
        self,
        client: ZhipuClient,
        vector_index: VectorMemoryIndex,
        cards: AdvancedJsonCardStore,
        embed_cache_path: str = os.path.join("memory_data", "embed_cache.json"),
    ):
        self.client = client
        self.embedder = ZhipuEmbedder(client)
        self.vector_index = vector_index
        self.cards = cards
        self.embed_cache_path = embed_cache_path

    def ask(self, question: str, top_k: int = 4) -> str:
        memories = self.vector_index.search(question, embedder=self.embedder, top_k=top_k)

        system_prompt = (
            "你是一个具备长期记忆的助理。你可以使用两类记忆：\n"
            "1) Advanced JSON Cards：结构化、相对稳定的核心事实；\n"
            "2) 检索到的历史对话块：包含上下文前缀与细节。\n\n"
            "请严格遵守：\n"
            "- 如信息冲突，以『更晚/更近』的对话指令为准（除非用户明确撤销）。\n"
            "- 如果缺少关键字段（金额/账户/时间等），先提问澄清，不要编造。\n\n"
            f"【结构化核心事实（JSON Cards）】\n{_format_cards(self.cards)}\n\n"
            f"【相关历史对话（向量检索 TopK）】\n{_format_memories(memories)}\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        return self.client.chat_completions(messages=messages)


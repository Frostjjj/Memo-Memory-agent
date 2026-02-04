import json
import os
import re
import hashlib
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fa5]+")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def infer_intent(text: str) -> str:
    if re.search(r"转账|电汇|支付|汇款", text):
        return "转账指令"
    if re.search(r"预订|订|机票|酒店|航班", text):
        return "预订"
    if re.search(r"确认|好的|就订|就这样", text):
        return "确认"
    if re.search(r"修改|更改|调整|改为", text):
        return "修改"
    if re.search(r"询问|怎么|为什么|能否", text):
        return "询问"
    return "交流"


@dataclass
class Turn:
    speaker: str
    text: str
    timestamp: str
    intent: str
    context_tag: Optional[str] = None


@dataclass
class MemoryChunk:
    chunk_id: str
    start_idx: int
    end_idx: int
    created_at: str
    prefix_summary: str
    turns: List[Turn]

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "created_at": self.created_at,
            "prefix_summary": self.prefix_summary,
            "turns": [asdict(t) for t in self.turns],
        }

    @staticmethod
    def from_dict(data: Dict) -> "MemoryChunk":
        return MemoryChunk(
            chunk_id=data["chunk_id"],
            start_idx=data["start_idx"],
            end_idx=data["end_idx"],
            created_at=data["created_at"],
            prefix_summary=data["prefix_summary"],
            turns=[Turn(**t) for t in data["turns"]],
        )


@dataclass
class AdvancedJsonCard:
    card_id: str
    facts: Dict
    updated_at: str


class AdvancedJsonCardStore:
    def __init__(self, cards: Optional[Dict[str, AdvancedJsonCard]] = None):
        self.cards = cards or {}

    def upsert(self, card_id: str, facts: Dict) -> None:
        self.cards[card_id] = AdvancedJsonCard(
            card_id=card_id,
            facts=facts,
            updated_at=_now_iso(),
        )

    def list_cards(self) -> List[AdvancedJsonCard]:
        return list(self.cards.values())

    def to_dict(self) -> Dict:
        return {
            "cards": [
                {"card_id": c.card_id, "facts": c.facts, "updated_at": c.updated_at}
                for c in self.cards.values()
            ]
        }

    @staticmethod
    def from_dict(data: Dict) -> "AdvancedJsonCardStore":
        cards = {}
        for c in data.get("cards", []):
            cards[c["card_id"]] = AdvancedJsonCard(
                card_id=c["card_id"],
                facts=c["facts"],
                updated_at=c["updated_at"],
            )
        return AdvancedJsonCardStore(cards)

    @classmethod
    def load(cls, path: str) -> "AdvancedJsonCardStore":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class ConversationMemory:
    def __init__(self, turns: Optional[List[Turn]] = None):
        self.turns = turns or []

    def add_turn(
        self,
        speaker: str,
        text: str,
        timestamp: Optional[str] = None,
        context_tag: Optional[str] = None,
    ) -> None:
        ts = timestamp or _now_iso()
        self.turns.append(
            Turn(
                speaker=speaker,
                text=text,
                timestamp=ts,
                intent=infer_intent(text),
                context_tag=context_tag,
            )
        )

    def to_dict(self) -> Dict:
        return {"turns": [asdict(t) for t in self.turns]}

    @staticmethod
    def from_dict(data: Dict) -> "ConversationMemory":
        return ConversationMemory(turns=[Turn(**t) for t in data.get("turns", [])])

    @classmethod
    def load(cls, path: str) -> "ConversationMemory":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


def _summarize_context(turns: List[Turn]) -> str:
    context_tags = [t.context_tag for t in turns if t.context_tag]
    if context_tags:
        return f"[上下文：{context_tags[-1]}]"

    time_range = f"{turns[0].timestamp} ~ {turns[-1].timestamp}"
    speakers = "、".join(sorted({t.speaker for t in turns}))
    intents = "、".join(sorted({t.intent for t in turns}))
    return f"[上下文：时间 {time_range}；人物 {speakers}；意图 {intents}]"


def _format_chunk_text(turns: List[Turn]) -> str:
    lines = [f"{t.speaker}：{t.text}" for t in turns]
    return "\n".join(lines)


def _format_chunk_text_compact(
    turns: List[Turn],
    *,
    max_turn_chars: int = 320,
    max_turns: int = 16,
    max_total_chars: int = 2400,
) -> str:
    """
    为 embedding / LLM 提示词准备的“紧凑版”对话文本：
    - 默认更偏向保留用户侧内容（更利于检索匹配 query）
    - 逐条截断，避免单条 input 过长触发云端限制
    - 限制最多保留的 turn 数，避免 chunk 内超长
    """
    # 思路：为避免“同一个 chunk 里较早的事实被裁掉”，采用 head+tail 覆盖：
    # - 先按原顺序生成每条 turn 的简化行（逐条截断）
    # - 如果超过 max_turns 或 max_total_chars，则保留一部分开头 + 一部分结尾
    lines_all: List[str] = []
    for t in turns:
        txt = (t.text or "").strip()
        if max_turn_chars > 0 and len(txt) > max_turn_chars:
            txt = txt[:max_turn_chars] + "…"
        lines_all.append(f"{t.speaker}：{txt}")

    # turn 数限制
    if max_turns > 0 and len(lines_all) > max_turns:
        head_n = max(1, max_turns // 2)
        tail_n = max(1, max_turns - head_n)
        lines_all = lines_all[:head_n] + ["…（中间省略）…"] + lines_all[-tail_n:]

    # 总字符预算限制（避免 embedding 单条 input 过长）
    def _join_len(ls: List[str]) -> int:
        return len("\n".join(ls))

    if max_total_chars > 0 and _join_len(lines_all) > max_total_chars:
        # 继续做 head+tail 收缩，直到满足预算或只剩很少行
        head = []
        tail = []
        i, j = 0, len(lines_all) - 1
        # 交替从头/尾取，尽量保留两端信息
        take_head = True
        while i <= j:
            candidate = (head + ["…（中间省略）…"] + list(reversed(tail))) if tail else head
            if max_total_chars > 0 and _join_len(candidate) >= max_total_chars and (head or tail):
                break
            if take_head:
                head.append(lines_all[i])
                i += 1
            else:
                tail.append(lines_all[j])
                j -= 1
            take_head = not take_head

        out = head
        if tail:
            out = head + ["…（中间省略）…"] + list(reversed(tail))
        # 兜底：如果仍然太长，直接硬截断整串
        s = "\n".join(out)
        if max_total_chars > 0 and len(s) > max_total_chars:
            s = s[:max_total_chars]
        return s

    return "\n".join(lines_all)


class MemoryIndex:
    def __init__(self, chunks: Optional[List[MemoryChunk]] = None):
        self.chunks = chunks or []
        self._idf_cache: Dict[str, float] = {}
        self._tf_cache: List[Dict[str, float]] = []

    def build(self, memory: ConversationMemory, chunk_size: int = 20) -> "MemoryIndex":
        chunks: List[MemoryChunk] = []
        total = len(memory.turns)
        chunk_id = 0
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            turns = memory.turns[start:end]
            prefix = _summarize_context(turns)
            chunks.append(
                MemoryChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    start_idx=start,
                    end_idx=end - 1,
                    created_at=_now_iso(),
                    prefix_summary=prefix,
                    turns=turns,
                )
            )
            chunk_id += 1
        self.chunks = chunks
        self._build_tfidf()
        return self

    def _build_tfidf(self) -> None:
        docs = []
        for chunk in self.chunks:
            doc = f"{chunk.prefix_summary}\n{_format_chunk_text(chunk.turns)}"
            docs.append(_tokenize(doc))

        df: Dict[str, int] = {}
        for tokens in docs:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        total_docs = max(len(docs), 1)
        self._idf_cache = {t: 1.0 + (total_docs / (1 + df_t)) for t, df_t in df.items()}
        self._tf_cache = []

        for tokens in docs:
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0
            self._tf_cache.append(tf)

    def to_dict(self) -> Dict:
        return {"chunks": [c.to_dict() for c in self.chunks]}

    @staticmethod
    def from_dict(data: Dict) -> "MemoryIndex":
        chunks = [MemoryChunk.from_dict(c) for c in data.get("chunks", [])]
        index = MemoryIndex(chunks=chunks)
        index._build_tfidf()
        return index

    @classmethod
    def load(cls, path: str) -> "MemoryIndex":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.chunks:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_tf: Dict[str, float] = {}
        for t in query_tokens:
            query_tf[t] = query_tf.get(t, 0.0) + 1.0

        def dot(a: Dict[str, float], b: Dict[str, float]) -> float:
            return sum(a.get(k, 0.0) * b.get(k, 0.0) for k in b.keys())

        def norm(a: Dict[str, float]) -> float:
            return sum(v * v for v in a.values()) ** 0.5

        scored: List[Tuple[float, int]] = []
        for idx, tf in enumerate(self._tf_cache):
            weighted_tf = {k: v * self._idf_cache.get(k, 1.0) for k, v in tf.items()}
            weighted_q = {k: v * self._idf_cache.get(k, 1.0) for k, v in query_tf.items()}
            score = dot(weighted_tf, weighted_q) / (norm(weighted_tf) * norm(weighted_q) + 1e-8)
            scored.append((score, idx))

        scored.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, idx in scored[:top_k]:
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "score": round(score, 4),
                    "prefix_summary": chunk.prefix_summary,
                    "text": _format_chunk_text(chunk.turns),
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                }
            )
        return results


def search_user_memory(query: str, index: MemoryIndex, top_k: int = 3) -> List[Dict]:
    return index.search(query, top_k=top_k)


class Embedder:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class LocalHashEmbedder(Embedder):
    """
    离线可用的“伪 embedding”（用于测试/无网络环境）。
    原理：将 token hash 到固定维度的稠密向量，再做 L2 归一化。
    这不是语义 embedding，但能验证向量检索链路是否工作。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vecs: List[List[float]] = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in _tokenize(text):
                h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            vecs.append(v.tolist())
        return vecs


class VectorMemoryIndex:
    """
    真正的 embedding 向量检索索引：
    - 索引阶段：对每个 chunk（prefix_summary + turns）做 embedding，并持久化向量
    - 应用阶段：对 query 做 embedding，计算余弦相似度取 top_k
    """

    def __init__(self, chunks: Optional[List[MemoryChunk]] = None, vectors: Optional[List[List[float]]] = None):
        self.chunks = chunks or []
        self.vectors = vectors or []
        # 兼容持久化：用于判断索引是否与当前 build 策略一致
        self.meta: Dict[str, str] = {}

    def build(
        self,
        memory: ConversationMemory,
        embedder: Embedder,
        chunk_size: int = 20,
        cache_path: Optional[str] = None,
    ) -> "VectorMemoryIndex":
        # 复用原本的分块/上下文前缀逻辑
        tmp = MemoryIndex().build(memory, chunk_size=chunk_size)
        self.chunks = tmp.chunks

        # 注意：我们始终保留完整 chunk 文本在 self.chunks 中（用于展示/回放），
        # 但用于 embedding 的文本采用“紧凑版”，这样聊天越久也不容易触发云端 input 限制。
        docs: List[str] = []
        for c in self.chunks:
            compact_turns = _format_chunk_text_compact(c.turns)
            docs.append(f"{c.prefix_summary}\n{compact_turns}")

        # 保护性处理：避免 embedding 请求过大（云端接口通常对请求体/单条 input 都有限制）
        # 说明：这里截断仅影响向量表示，不影响前端展示的 chunk 文本。
        max_doc_chars_env = os.getenv("EMBED_DOC_MAX_CHARS", "").strip()
        try:
            max_doc_chars = int(max_doc_chars_env) if max_doc_chars_env else 8000
        except ValueError:
            max_doc_chars = 8000

        def _prep_for_embedding(s: str) -> str:
            s = (s or "").strip()
            if max_doc_chars > 0 and len(s) > max_doc_chars:
                return s[:max_doc_chars]
            return s

        embed_docs: List[str] = [_prep_for_embedding(d) for d in docs]
        self.meta = {
            "embed_doc_format": "compact_v1",
            "embed_doc_max_chars": str(max_doc_chars),
        }

        cache: Dict[str, List[float]] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f).get("embeddings", {})

        keys: List[str] = []
        missing_docs: List[str] = []
        missing_keys: List[str] = []
        for doc in embed_docs:
            key = hashlib.sha256(doc.encode("utf-8")).hexdigest()
            keys.append(key)
            if key not in cache:
                missing_docs.append(doc)
                missing_keys.append(key)

        if missing_docs:
            # 分批调用 embedding，避免单次请求体过大导致 400 Request Entity Too Large
            max_batch_texts_env = os.getenv("EMBED_MAX_BATCH_TEXTS", "").strip()
            max_batch_chars_env = os.getenv("EMBED_MAX_BATCH_CHARS", "").strip()
            try:
                max_batch_texts = int(max_batch_texts_env) if max_batch_texts_env else 16
            except ValueError:
                max_batch_texts = 16
            try:
                max_batch_chars = int(max_batch_chars_env) if max_batch_chars_env else 20000
            except ValueError:
                max_batch_chars = 20000

            def _too_large_error(e: Exception) -> bool:
                msg = str(e)
                return ("Request Entity Too Large" in msg) or ("1210" in msg)

            def _iter_batches(texts: List[str]) -> List[List[str]]:
                batches: List[List[str]] = []
                cur: List[str] = []
                cur_chars = 0
                for t in texts:
                    t_chars = len(t)
                    # 单条就超，也至少单独发（后续会走自适应拆分/报错）
                    if cur and (
                        (max_batch_texts > 0 and len(cur) >= max_batch_texts)
                        or (max_batch_chars > 0 and cur_chars + t_chars > max_batch_chars)
                    ):
                        batches.append(cur)
                        cur = []
                        cur_chars = 0
                    cur.append(t)
                    cur_chars += t_chars
                if cur:
                    batches.append(cur)
                return batches

            def _embed_adaptive(batch: List[str]) -> List[List[float]]:
                try:
                    return embedder.embed_texts(batch)
                except Exception as e:
                    if _too_large_error(e):
                        # 如果是多条 input，递归二分拆小批次
                        if len(batch) > 1:
                            mid = len(batch) // 2
                            return _embed_adaptive(batch[:mid]) + _embed_adaptive(batch[mid:])

                        # 如果单条 input 仍然过大，说明单条文本本身超过模型/网关限制：
                        # 继续做保护性截断并重试（直到成功或截断到很小）
                        if len(batch) == 1:
                            t = batch[0]
                            min_chars_env = os.getenv("EMBED_DOC_MIN_CHARS", "").strip()
                            try:
                                min_chars = int(min_chars_env) if min_chars_env else 400
                            except ValueError:
                                min_chars = 400
                            cur = t
                            while len(cur) > max(1, min_chars):
                                cur = cur[: max(min_chars, len(cur) // 2)]
                                try:
                                    return embedder.embed_texts([cur])
                                except Exception as e2:
                                    if not _too_large_error(e2):
                                        raise
                            # 仍然失败：抛出原始异常，便于定位
                    raise

            new_vecs_all: List[List[float]] = []
            for batch in _iter_batches(missing_docs):
                new_vecs_all.extend(_embed_adaptive(batch))

            if len(new_vecs_all) != len(missing_docs):
                raise RuntimeError(
                    f"embedding 数量不匹配：期望 {len(missing_docs)}，实际 {len(new_vecs_all)}"
                )
            for k, v in zip(missing_keys, new_vecs_all):
                cache[k] = v

        self.vectors = [cache[k] for k in keys]

        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"embeddings": cache}, f, ensure_ascii=False)

        return self

    def to_dict(self) -> Dict:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self.vectors,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(data: Dict) -> "VectorMemoryIndex":
        chunks = [MemoryChunk.from_dict(c) for c in data.get("chunks", [])]
        vectors = data.get("vectors", [])
        idx = VectorMemoryIndex(chunks=chunks, vectors=vectors)
        idx.meta = data.get("meta", {}) or {}
        return idx

    @classmethod
    def load(cls, path: str) -> "VectorMemoryIndex":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def search(self, query: str, embedder: Embedder, top_k: int = 3) -> List[Dict]:
        if not self.chunks or not self.vectors:
            return []

        qvec = embedder.embed_texts([query])[0]
        scored: List[Tuple[float, int]] = []
        for i, v in enumerate(self.vectors):
            scored.append((self._cosine(qvec, v), i))
        scored.sort(reverse=True, key=lambda x: x[0])

        # 关键词兜底：向量检索可能因为截断/表达差异漏召回，简单的 token overlap 能显著提升“很久以前说过的原词”命中率
        q_tokens = set(_tokenize(query))
        keyword_scores: List[Tuple[float, int]] = []
        if q_tokens:
            for i, c in enumerate(self.chunks):
                doc = f"{c.prefix_summary}\n{_format_chunk_text(c.turns)}"
                dt = set(_tokenize(doc))
                if not dt:
                    continue
                overlap = len(q_tokens & dt) / max(1, len(q_tokens))
                if overlap > 0:
                    keyword_scores.append((overlap, i))
            keyword_scores.sort(reverse=True, key=lambda x: x[0])

        # 合并：先拿向量 top_k，再用关键词补齐（去重）
        picked: List[Tuple[float, int, str]] = []  # (score, idx, source)
        for score, idx in scored[: max(1, top_k)]:
            picked.append((score, idx, "vec"))
        if len(picked) < max(1, top_k):
            for ks, idx in keyword_scores:
                if any(p[1] == idx for p in picked):
                    continue
                picked.append((ks, idx, "kw"))
                if len(picked) >= top_k:
                    break

        results: List[Dict] = []
        for score, idx, src in picked[:top_k]:
            c = self.chunks[idx]
            results.append(
                {
                    "chunk_id": c.chunk_id,
                    "score": round(float(score), 4),
                    "source": src,
                    "prefix_summary": c.prefix_summary,
                    # 保留完整文本便于前端查看，同时提供 compact 版给上层拼 prompt 用
                    "text": _format_chunk_text(c.turns),
                    "text_compact": _format_chunk_text_compact(c.turns),
                    "start_idx": c.start_idx,
                    "end_idx": c.end_idx,
                }
            )
        return results

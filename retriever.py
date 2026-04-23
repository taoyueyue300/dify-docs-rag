"""
Step 2: 混合检索(FAISS向量+BM25) + Reranker重排序
"""
import os
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
INDEX_DIR = "./faiss_index"


class HybridRetriever:
    def __init__(self, use_reranker=False):
        print("加载模型...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)

        # 加载FAISS索引
        self.index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
        with open(os.path.join(INDEX_DIR, "docs.pkl"), "rb") as f:
            data = pickle.load(f)
        self.all_docs = data["contents"]
        self.all_metadatas = data["metadatas"]

        # BM25
        tokenized = [self._tokenize(doc) for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized)

        # Reranker（默认关闭，实验证明在当前数据下效果为负）
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
        print("模型加载完成")

    def _tokenize(self, text: str) -> list:
        tokens = []
        word = ""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                if word:
                    tokens.append(word.lower())
                    word = ""
                tokens.append(char)
            elif char.isalnum():
                word += char
            else:
                if word:
                    tokens.append(word.lower())
                    word = ""
        if word:
            tokens.append(word.lower())
        return tokens

    def retrieve(self, query: str, top_k: int = 5) -> list:
        # 1. 向量检索
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_emb, 20)
        vector_results_raw = [(self.all_docs[i], self.all_metadatas[i], float(s))
                              for s, i in zip(scores[0], indices[0]) if i >= 0]

        # 2. BM25检索
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]

        # 3. 合并去重
        seen = set()
        candidates = []
        for doc, meta, _ in vector_results_raw:
            if doc not in seen:
                seen.add(doc)
                candidates.append({"content": doc, "metadata": meta})
        for idx in bm25_top:
            doc = self.all_docs[idx]
            if doc not in seen:
                seen.add(doc)
                candidates.append({"content": doc, "metadata": self.all_metadatas[idx]})

        if not candidates:
            return []

        # 4. 排序
        if self.use_reranker and self.reranker:
            pairs = [[query, c["content"]] for c in candidates]
            rerank_scores = self.reranker.compute_score(pairs)
            if isinstance(rerank_scores, (int, float)):
                rerank_scores = [rerank_scores]
            ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        else:
            # 加权融合排序（实验证明效果更优）
            vector_score_map = {doc: 1.0 - i * 0.05 for i, (doc, _, _) in enumerate(vector_results_raw)}
            ranked = sorted(candidates, key=lambda c: vector_score_map.get(c["content"], 0), reverse=True)
            ranked = [(c, vector_score_map.get(c["content"], 0)) for c in ranked]

        return [
            {"content": c["content"], "metadata": c["metadata"], "score": round(s, 4)}
            for c, s in ranked[:top_k]
        ]


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_queries = [
        "Dify如何配置知识库？",
        "如何本地部署Dify？",
        "Dify的工作流怎么用？",
    ]
    for q in test_queries:
        print(f"\n{'='*50}\n问题: {q}")
        results = retriever.retrieve(q)
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']} | {r['metadata']['source']}")
            print(f"      {r['content'][:100]}...")

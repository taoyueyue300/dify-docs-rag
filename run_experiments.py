"""
自动化评测对比实验
- 对比不同 chunk_size (300, 500, 800)
- 对比不同 BM25/向量权重
- 对比有无 Reranker
"""
import os
import sys
import json
import pickle
import gc
import time
import glob
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np

load_dotenv()

DOCS_PATH = os.getenv("DOCS_PATH", "../dify/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8317/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "my-api-key-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3-flash-preview")

# ========== 测试集 ==========
TEST_SET = [
    {
        "question": "如何本地部署Dify？",
        "keywords": ["docker", "compose", "部署", "安装", "CPU", "RAM"],
    },
    {
        "question": "Dify支持哪些模型？",
        "keywords": ["GPT", "Llama", "模型", "OpenAI", "开源"],
    },
    {
        "question": "Dify的RAG知识库怎么用？",
        "keywords": ["知识库", "RAG", "文档", "检索", "向量"],
    },
    {
        "question": "如何配置Dify的环境变量？",
        "keywords": [".env", "配置", "环境", "变量", "docker"],
    },
    {
        "question": "Dify的API如何使用？",
        "keywords": ["API", "接口", "调用", "token", "请求"],
    },
    {
        "question": "Dify支持哪些向量数据库？",
        "keywords": ["向量", "Weaviate", "Qdrant", "Milvus", "数据库"],
    },
    {
        "question": "Dify如何创建工作流应用？",
        "keywords": ["工作流", "workflow", "节点", "应用"],
    },
    {
        "question": "Dify的系统架构是什么？",
        "keywords": ["架构", "前端", "后端", "API", "Flask", "Next"],
    },
    {
        "question": "如何为Dify贡献代码？",
        "keywords": ["贡献", "PR", "pull request", "开发", "fork"],
    },
    {
        "question": "Dify支持哪些文件格式的知识库导入？",
        "keywords": ["PDF", "文件", "格式", "导入", "Markdown", "txt"],
    },
]


# ========== 工具函数 ==========

def collect_files():
    base = str(Path(DOCS_PATH).resolve())
    all_files = glob.glob(os.path.join(base, "**/*.md"), recursive=True)
    excluded = ["node_modules", ".venv", "dist", "build", "__pycache__", "plugin_daemon"]
    return [f for f in all_files if not any(ex in f for ex in excluded)]


def load_docs(file_paths):
    documents = []
    base = Path(DOCS_PATH).resolve()
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content.strip()) < 50:
                continue
            rel_path = str(Path(path).relative_to(base))
            documents.append({"content": content, "source": rel_path})
        except:
            pass
    return documents


def split_text(text, chunk_size, overlap=50):
    separators = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "]
    chunks = []

    def _split(text, sep_idx=0):
        if len(text) <= chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        if sep_idx >= len(separators):
            for i in range(0, len(text), chunk_size - overlap):
                piece = text[i:i + chunk_size]
                if piece.strip():
                    chunks.append(piece.strip())
            return
        sep = separators[sep_idx]
        parts = text.split(sep)
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    _split(current, sep_idx + 1)
                current = part
        if current:
            _split(current, sep_idx + 1)

    _split(text)
    return chunks


def tokenize(text):
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


# ========== 核心实验逻辑 ==========

def build_index(documents, chunk_size, embed_model):
    """构建索引，返回 (faiss_index, contents, metadatas, embeddings)"""
    all_chunks = []
    for doc in documents:
        chunks = split_text(doc["content"], chunk_size)
        for chunk in chunks:
            all_chunks.append({"content": chunk, "source": doc["source"]})

    contents = [c["content"] for c in all_chunks]
    metadatas = [{"source": c["source"]} for c in all_chunks]

    embeddings = embed_model.encode(contents, normalize_embeddings=True, batch_size=16, show_progress_bar=False)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    return index, contents, metadatas, embeddings


def retrieve_hybrid(query, embed_model, index, contents, metadatas, bm25,
                    top_k=5, vector_weight=0.6, use_reranker=True, reranker=None):
    """执行混合检索"""
    # 向量检索
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, 20)
    vector_results = [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]

    # BM25
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]

    # 合并去重
    seen = set()
    candidates = []
    for idx, _ in vector_results:
        if idx not in seen:
            seen.add(idx)
            candidates.append(idx)
    for idx in bm25_top:
        if idx not in seen:
            seen.add(idx)
            candidates.append(idx)

    if not candidates:
        return []

    if use_reranker and reranker:
        pairs = [[query, contents[i]] for i in candidates]
        rerank_scores = reranker.compute_score(pairs)
        if isinstance(rerank_scores, (int, float)):
            rerank_scores = [rerank_scores]
        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        return [{"content": contents[i], "metadata": metadatas[i], "score": s} for i, s in ranked[:top_k]]
    else:
        # 无Reranker：用加权分数
        # 归一化vector scores
        max_vs = max(s for _, s in vector_results) if vector_results else 1
        vector_score_map = {i: s / max_vs for i, s in vector_results}
        # 归一化bm25 scores
        max_bs = max(bm25_scores[i] for i in bm25_top) if bm25_top else 1
        bm25_score_map = {i: bm25_scores[i] / max_bs for i in bm25_top}

        scored = []
        for idx in candidates:
            vs = vector_score_map.get(idx, 0) * vector_weight
            bs = bm25_score_map.get(idx, 0) * (1 - vector_weight)
            scored.append((idx, vs + bs))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{"content": contents[i], "metadata": metadatas[i], "score": s} for i, s in scored[:top_k]]


def evaluate_retrieval(results_per_query, test_set):
    """评估检索命中率"""
    hits = 0
    details = []
    for item, results in zip(test_set, results_per_query):
        retrieved_text = " ".join([r["content"] for r in results]).lower()
        matched = [kw for kw in item["keywords"] if kw.lower() in retrieved_text]
        ratio = len(matched) / len(item["keywords"])
        hit = ratio >= 0.3
        if hit:
            hits += 1
        details.append({
            "question": item["question"],
            "keyword_hit_ratio": round(ratio, 3),
            "matched_keywords": matched,
            "hit": hit,
        })
    return {
        "hit_rate": round(hits / len(test_set), 3),
        "hits": hits,
        "total": len(test_set),
        "details": details,
    }


def run_experiments():
    print("=" * 70)
    print("  Dify文档RAG系统 — 自动化对比评测实验")
    print("=" * 70)

    # 加载数据
    print("\n[1/5] 加载文档...")
    file_paths = collect_files()
    documents = load_docs(file_paths)
    print(f"  共 {len(documents)} 个文档")

    # 加载模型
    print("\n[2/5] 加载Embedding模型...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    print("\n[3/5] 加载Reranker模型...")
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)

    all_results = {}

    # ==========================================
    # 实验1：不同 chunk_size
    # ==========================================
    print("\n" + "=" * 70)
    print("  实验1：chunk_size 对比 (300 vs 500 vs 800)")
    print("=" * 70)

    for chunk_size in [300, 500, 800]:
        print(f"\n  >> chunk_size = {chunk_size}")
        index, contents, metadatas, _ = build_index(documents, chunk_size, embed_model)
        print(f"     chunk数量: {len(contents)}")

        tokenized = [tokenize(doc) for doc in contents]
        bm25 = BM25Okapi(tokenized)

        results_per_query = []
        for item in TEST_SET:
            results = retrieve_hybrid(
                item["question"], embed_model, index, contents, metadatas, bm25,
                use_reranker=True, reranker=reranker
            )
            results_per_query.append(results)

        eval_result = evaluate_retrieval(results_per_query, TEST_SET)
        key = f"chunk_size={chunk_size}"
        all_results[key] = eval_result
        print(f"     命中率: {eval_result['hit_rate']} ({eval_result['hits']}/{eval_result['total']})")

    # ==========================================
    # 实验2：不同 BM25/向量权重（使用 chunk_size=500）
    # ==========================================
    print("\n" + "=" * 70)
    print("  实验2：BM25/向量权重对比（chunk_size=500, 有Reranker）")
    print("=" * 70)

    index, contents, metadatas, _ = build_index(documents, 500, embed_model)
    tokenized = [tokenize(doc) for doc in contents]
    bm25 = BM25Okapi(tokenized)

    for vector_w in [0.3, 0.5, 0.7, 0.9]:
        print(f"\n  >> 向量权重={vector_w}, BM25权重={round(1-vector_w, 1)}")
        results_per_query = []
        for item in TEST_SET:
            results = retrieve_hybrid(
                item["question"], embed_model, index, contents, metadatas, bm25,
                vector_weight=vector_w, use_reranker=True, reranker=reranker
            )
            results_per_query.append(results)

        eval_result = evaluate_retrieval(results_per_query, TEST_SET)
        key = f"vector_w={vector_w}_bm25_w={round(1-vector_w,1)}"
        all_results[key] = eval_result
        print(f"     命中率: {eval_result['hit_rate']} ({eval_result['hits']}/{eval_result['total']})")

    # ==========================================
    # 实验3：有无 Reranker 对比
    # ==========================================
    print("\n" + "=" * 70)
    print("  实验3：有无 Reranker 对比（chunk_size=500, 向量权重=0.6）")
    print("=" * 70)

    for use_rr, label in [(True, "有Reranker"), (False, "无Reranker")]:
        print(f"\n  >> {label}")
        results_per_query = []
        for item in TEST_SET:
            results = retrieve_hybrid(
                item["question"], embed_model, index, contents, metadatas, bm25,
                vector_weight=0.6, use_reranker=use_rr, reranker=reranker
            )
            results_per_query.append(results)

        eval_result = evaluate_retrieval(results_per_query, TEST_SET)
        key = label
        all_results[key] = eval_result
        print(f"     命中率: {eval_result['hit_rate']} ({eval_result['hits']}/{eval_result['total']})")

    # ==========================================
    # 汇总报告
    # ==========================================
    print("\n" + "=" * 70)
    print("  汇总报告")
    print("=" * 70)

    print(f"\n{'配置':<35} | {'命中率':<8} | {'命中/总数'}")
    print("-" * 65)
    for key, result in all_results.items():
        print(f"  {key:<33} | {result['hit_rate']:<8} | {result['hits']}/{result['total']}")

    # 保存结果
    with open("experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 生成Markdown报告
    with open("experiment_report.md", "w", encoding="utf-8") as f:
        f.write("# RAG系统评测对比实验报告\n\n")
        f.write(f"评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"文档数量: {len(documents)} | Embedding模型: {EMBEDDING_MODEL}\n\n")

        f.write("## 实验1：chunk_size 对比\n\n")
        f.write("| chunk_size | chunk数量 | 命中率 |\n")
        f.write("|:---:|:---:|:---:|\n")
        for cs in [300, 500, 800]:
            key = f"chunk_size={cs}"
            r = all_results[key]
            f.write(f"| {cs} | - | {r['hit_rate']} ({r['hits']}/{r['total']}) |\n")

        f.write("\n## 实验2：检索权重对比\n\n")
        f.write("| 向量权重 | BM25权重 | 命中率 |\n")
        f.write("|:---:|:---:|:---:|\n")
        for vw in [0.3, 0.5, 0.7, 0.9]:
            key = f"vector_w={vw}_bm25_w={round(1-vw,1)}"
            r = all_results[key]
            f.write(f"| {vw} | {round(1-vw,1)} | {r['hit_rate']} ({r['hits']}/{r['total']}) |\n")

        f.write("\n## 实验3：Reranker 效果对比\n\n")
        f.write("| 配置 | 命中率 |\n")
        f.write("|:---:|:---:|\n")
        for label in ["有Reranker", "无Reranker"]:
            r = all_results[label]
            f.write(f"| {label} | {r['hit_rate']} ({r['hits']}/{r['total']}) |\n")

        f.write("\n## 结论\n\n")
        # 找最优chunk_size
        best_cs = max([300, 500, 800], key=lambda x: all_results[f"chunk_size={x}"]["hit_rate"])
        f.write(f"- 最优 chunk_size: **{best_cs}**\n")
        # 找最优权重
        best_vw = max([0.3, 0.5, 0.7, 0.9],
                      key=lambda x: all_results[f"vector_w={x}_bm25_w={round(1-x,1)}"]["hit_rate"])
        f.write(f"- 最优检索权重: 向量 **{best_vw}** / BM25 **{round(1-best_vw,1)}**\n")
        # Reranker对比
        rr_gain = all_results["有Reranker"]["hit_rate"] - all_results["无Reranker"]["hit_rate"]
        f.write(f"- Reranker 提升: **+{rr_gain:.1%}** 命中率\n")

    print(f"\n报告已保存:")
    print(f"  - experiment_report.json")
    print(f"  - experiment_report.md")
    print("\n实验完成!")


if __name__ == "__main__":
    run_experiments()

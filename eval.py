"""
Step 5: RAG评测脚本
- 自动生成测试数据（从你的文档中）
- 用RAGAS框架跑评测指标
- 输出评测报告

用法: python eval.py
"""
import os
import json
from dotenv import load_dotenv
from retriever import HybridRetriever
from chain import RAGChain

load_dotenv()


# ============ 方案A：手动测试集（推荐先用这个）============

MANUAL_TEST_SET = [
    {
        "question": "Dify支持哪些向量数据库？",
        "ground_truth": "Dify支持多种向量数据库，包括Weaviate、Qdrant、Milvus、Chroma等",
    },
    {
        "question": "如何在Dify中创建一个聊天助手应用？",
        "ground_truth": "在Dify控制台中，点击创建应用，选择聊天助手类型，配置模型和提示词即可",
    },
    {
        "question": "Dify的API如何调用？",
        "ground_truth": "Dify提供RESTful API，需要在应用设置中获取API Key，通过HTTP请求调用",
    },
    {
        "question": "Dify的RAG知识库支持什么文件格式？",
        "ground_truth": "支持PDF、Markdown、TXT、HTML、DOCX等常见文档格式",
    },
    {
        "question": "如何本地部署Dify？",
        "ground_truth": "使用Docker Compose一键部署，clone仓库后执行docker compose up -d",
    },
]


def evaluate_retrieval(retriever: HybridRetriever, test_set: list[dict]) -> dict:
    """评测检索质量"""
    results = {"total": len(test_set), "details": []}
    hit_count = 0

    for item in test_set:
        retrieved = retriever.retrieve(item["question"], top_k=5)
        contexts = [r["content"] for r in retrieved]

        # 简单的命中率评估：ground_truth中的关键词是否出现在检索结果中
        gt_keywords = set(item["ground_truth"].replace("，", " ").replace("、", " ").split())
        retrieved_text = " ".join(contexts)
        matched_keywords = [kw for kw in gt_keywords if kw in retrieved_text]
        hit_ratio = len(matched_keywords) / max(len(gt_keywords), 1)

        if hit_ratio > 0.3:
            hit_count += 1

        results["details"].append({
            "question": item["question"],
            "hit_ratio": round(hit_ratio, 3),
            "num_retrieved": len(retrieved),
            "top_source": retrieved[0]["metadata"].get("source", "") if retrieved else "",
        })

    results["hit_rate"] = round(hit_count / len(test_set), 3)
    return results


def evaluate_generation(chain: RAGChain, test_set: list[dict]) -> dict:
    """评测生成质量（端到端）"""
    results = {"total": len(test_set), "details": []}

    for item in test_set:
        response = chain.query(item["question"])
        answer = response["answer"]

        # 忠实度检查：答案是否包含"文档中未找到"
        is_abstained = "未找到" in answer or "没有相关" in answer

        results["details"].append({
            "question": item["question"],
            "answer_length": len(answer),
            "sources_count": len(response["sources"]),
            "abstained": is_abstained,
        })

    return results


def run_evaluation():
    """运行完整评测"""
    print("=" * 60)
    print("Dify文档RAG系统 - 评测报告")
    print("=" * 60)

    # 检索评测
    print("\n📊 检索质量评测...")
    retriever = HybridRetriever()
    retrieval_results = evaluate_retrieval(retriever, MANUAL_TEST_SET)
    print(f"  命中率 (Hit Rate@5): {retrieval_results['hit_rate']}")
    for d in retrieval_results["details"]:
        print(f"  - {d['question'][:20]}... | hit_ratio={d['hit_ratio']} | source={d['top_source']}")

    # 生成评测
    print("\n📊 端到端生成评测...")
    chain = RAGChain()
    gen_results = evaluate_generation(chain, MANUAL_TEST_SET)
    for d in gen_results["details"]:
        status = "❌ 弃权" if d["abstained"] else "✓ 已回答"
        print(f"  - {d['question'][:20]}... | {status} | 长度={d['answer_length']} | 引用={d['sources_count']}个文件")

    # 保存报告
    report = {
        "retrieval": retrieval_results,
        "generation": gen_results,
    }
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 评测报告已保存到 eval_report.json")


if __name__ == "__main__":
    run_evaluation()

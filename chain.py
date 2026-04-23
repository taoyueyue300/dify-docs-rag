"""
Step 3: RAG问答链 — 检索 + LLM生成（使用OpenAI兼容API）
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from retriever import HybridRetriever

load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8317/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "my-api-key-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3-flash-preview")

SYSTEM_PROMPT = """你是Dify平台的技术助手。根据提供的文档片段回答用户问题。

规则：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确说"文档中未找到相关信息"
3. 回答末尾标注引用来源（文件路径）
4. 使用中文回答"""


class RAGChain:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.history = []

    def query(self, question: str) -> dict:
        """检索 → 组装prompt → LLM生成"""
        results = self.retriever.retrieve(question, top_k=5)

        context_parts = []
        sources = []
        for i, r in enumerate(results):
            src = r["metadata"].get("source", "unknown")
            context_parts.append(f"[文档{i+1}] (来源: {src})\n{r['content']}")
            sources.append(src)

        context = "\n\n".join(context_parts)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in self.history[-6:]:
            messages.append(h)
        messages.append({
            "role": "user",
            "content": f"参考文档：\n{context}\n\n用户问题：{question}",
        })

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": list(set(sources)),
            "retrieved_chunks": len(results),
        }


if __name__ == "__main__":
    chain = RAGChain()
    print("Dify文档助手已启动（输入 quit 退出）\n")
    while True:
        q = input("你: ")
        if q.strip().lower() in ("quit", "exit", "q"):
            break
        result = chain.query(q)
        print(f"\n助手: {result['answer']}")
        print(f"引用: {result['sources']}\n")

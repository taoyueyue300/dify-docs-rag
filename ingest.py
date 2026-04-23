"""
Step 1: 文档加载 → 分块 → 向量化 → 存入FAISS
用法: python ingest.py
"""
import os
import glob
import json
import pickle
import gc
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()

DOCS_PATH = os.getenv("DOCS_PATH", "../dify/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
INDEX_DIR = "./faiss_index"


def collect_markdown_files(base_path: str) -> list:
    all_files = glob.glob(os.path.join(base_path, "**/*.md"), recursive=True)
    excluded = ["node_modules", ".venv", "dist", "build", "__pycache__", "plugin_daemon"]
    filtered = [f for f in all_files if not any(ex in f for ex in excluded)]
    print(f"找到 {len(filtered)} 个Markdown文件")
    return filtered


def load_documents(file_paths: list) -> list:
    documents = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content.strip()) < 50:
                continue
            rel_path = str(Path(path).relative_to(Path(DOCS_PATH).resolve()))
            documents.append({"content": content, "source": rel_path})
        except Exception as e:
            print(f"跳过: {path} ({e})")
    print(f"成功加载 {len(documents)} 个文档")
    return documents


def split_text(text: str, chunk_size=800, overlap=80) -> list:
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


def split_documents(documents: list) -> list:
    all_chunks = []
    for doc in documents:
        chunks = split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append({"content": chunk, "source": doc["source"]})
    print(f"分块后共 {len(all_chunks)} 个chunk")
    return all_chunks


def create_index(chunks: list):
    os.makedirs(INDEX_DIR, exist_ok=True)

    print(f"加载Embedding模型 {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    contents = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    print("正在计算向量...")
    embeddings = model.encode(contents, show_progress_bar=True, normalize_embeddings=True, batch_size=16)

    del model
    gc.collect()

    # 创建FAISS索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积（因为已normalize，等价于余弦）
    index.add(embeddings.astype(np.float32))

    # 保存
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "docs.pkl"), "wb") as f:
        pickle.dump({"contents": contents, "metadatas": metadatas}, f)

    print(f"索引已保存到 {INDEX_DIR}，共 {index.ntotal} 条记录")


def main():
    file_paths = collect_markdown_files(str(Path(DOCS_PATH).resolve()))
    documents = load_documents(file_paths)
    chunks = split_documents(documents)
    create_index(chunks)
    print("文档入库完成")


if __name__ == "__main__":
    main()

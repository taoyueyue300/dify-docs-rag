"""
增强版 ingest 脚本：在原 ingest.py 基础上扩展支持 PDF / DOCX / HTML / 在线 URL。
不改动原 ingest.py（保持简单 baseline 可对照），新逻辑全部走 loaders.py。

用法：
    # 索引单一目录下所有支持的格式（md/pdf/docx/html/txt）
    python ingest_multi.py --path ./data --exts .md .pdf .docx

    # 索引若干在线网页
    python ingest_multi.py --urls https://docs.dify.ai/zh-hans/introduction https://example.com/blog/x.html

    # 混合：本地目录 + 网页 + 单 PDF
    python ingest_multi.py --path ./data --urls https://x.com/y.html --files ./extra.pdf

依赖：pip install pypdf python-docx beautifulsoup4 lxml httpx
"""
from __future__ import annotations

import argparse
import os
import gc
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from loaders import collect_files, load_any
from ingest import split_text  # 复用既有切分函数

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
INDEX_DIR = os.getenv("INDEX_DIR", "./faiss_index_multi")


def build_documents(paths: list[str]) -> list[dict]:
    """统一 loader：每个 path（或 URL）返回 {content, source, kind}。"""
    docs = []
    for p in paths:
        text = load_any(p)
        if not text or len(text.strip()) < 50:
            continue
        kind = "url" if p.startswith(("http://", "https://")) else Path(p).suffix.lower().lstrip(".")
        # source：URL 直接用，本地路径用相对路径（找不到 base 时退回绝对路径）
        try:
            source = str(Path(p).resolve())
        except Exception:
            source = p
        docs.append({"content": text, "source": source, "kind": kind})
    print(f"loaded {len(docs)} documents (kinds: "
          f"{ {k: sum(1 for d in docs if d['kind']==k) for k in set(d['kind'] for d in docs)} })")
    return docs


def split_documents(docs: list[dict]) -> list[dict]:
    chunks = []
    for d in docs:
        for c in split_text(d["content"]):
            chunks.append({"content": c, "source": d["source"], "kind": d["kind"]})
    print(f"chunks: {len(chunks)}")
    return chunks


def build_index(chunks: list[dict], index_dir: str = INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)
    print(f"loading embedding model {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    contents = [c["content"] for c in chunks]
    metadatas = [{"source": c["source"], "kind": c["kind"]} for c in chunks]
    print("encoding ...")
    embeddings = model.encode(contents, show_progress_bar=True,
                              normalize_embeddings=True, batch_size=16)
    del model
    gc.collect()

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "docs.pkl"), "wb") as f:
        pickle.dump({"contents": contents, "metadatas": metadatas}, f)
    print(f"OK saved to {index_dir} (n={index.ntotal})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="", help="recursive scan directory")
    ap.add_argument("--exts", nargs="+", default=[".md", ".pdf", ".docx", ".html", ".txt"])
    ap.add_argument("--files", nargs="+", default=[], help="extra individual files")
    ap.add_argument("--urls", nargs="+", default=[], help="online urls")
    ap.add_argument("--index-dir", default=INDEX_DIR)
    args = ap.parse_args()

    paths: list[str] = []
    if args.path:
        paths.extend(collect_files(args.path, exts=tuple(args.exts)))
    paths.extend(args.files)
    paths.extend(args.urls)

    if not paths:
        ap.error("no paths/urls provided")

    docs = build_documents(paths)
    chunks = split_documents(docs)
    build_index(chunks, args.index_dir)


if __name__ == "__main__":
    main()

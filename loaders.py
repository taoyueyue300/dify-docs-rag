"""
多格式文档加载器
支持: .md / .pdf / .html / .docx / .txt 以及在线 URL（含网页 / RSS）

设计原则：
1. 每个 loader 只关心"把字节流变成 plain text"，下游 chunk/embed 共用
2. 失败不抛异常，返回 None 由调用方过滤，避免单个文件挂掉整批
3. 网页内容统一抽正文（去除导航/页脚），减少噪声 chunk
"""
from __future__ import annotations

import os
import re
import io
from pathlib import Path
from urllib.parse import urlparse


# ---------- text-based ----------

def load_markdown(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[md] skip {path}: {e}")
        return None


def load_txt(path: str) -> str | None:
    return load_markdown(path)


# ---------- pdf ----------

def load_pdf(path: str) -> str | None:
    """优先用 pypdf；扫描版/复杂排版建议 pdfplumber 或 marker-pdf。
    依赖: pip install pypdf
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("[pdf] pypdf 未安装，请: pip install pypdf")
        return None
    try:
        reader = PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r"\n{3,}", "\n\n", text)
            if text.strip():
                # 给每页加锚点，方便后续按 chunk 回溯出处
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        print(f"[pdf] skip {path}: {e}")
        return None


# ---------- docx ----------

def load_docx(path: str) -> str | None:
    """依赖: pip install python-docx"""
    try:
        import docx
    except ImportError:
        print("[docx] python-docx 未安装，请: pip install python-docx")
        return None
    try:
        d = docx.Document(path)
        paras = [p.text for p in d.paragraphs if p.text.strip()]
        return "\n\n".join(paras) if paras else None
    except Exception as e:
        print(f"[docx] skip {path}: {e}")
        return None


# ---------- html (本地或下载后的) ----------

def load_html(html_str: str, base_url: str = "") -> str | None:
    """从 HTML 字符串抽取正文。
    依赖: pip install beautifulsoup4 lxml
    策略：① 先尝试 <article> / <main>；② 否则去掉 nav/footer/script/style 后取 body；
    ③ 折叠多余空白；保留小标题（h1-h3）。
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("[html] bs4 未安装，请: pip install beautifulsoup4 lxml")
        return None
    try:
        soup = BeautifulSoup(html_str, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "aside", "form", "noscript"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body
        if not main:
            return None
        # 把 h1-h3 转成 markdown 风格小标题，利于后续按标题切分
        for level in (1, 2, 3):
            for h in main.find_all(f"h{level}"):
                h.insert_before("\n" + "#" * level + " ")
                h.insert_after("\n")
        text = main.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text if len(text) > 50 else None
    except Exception as e:
        print(f"[html] parse failed: {e}")
        return None


def load_html_file(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return load_html(f.read())
    except Exception as e:
        print(f"[html] skip {path}: {e}")
        return None


# ---------- 在线 URL（网页 / RSS / 直链 PDF）----------

def load_url(url: str, timeout: float = 15.0) -> str | None:
    """支持 http(s)://.../something.{html,pdf}（按 Content-Type 自动 dispatch）。
    依赖: pip install httpx
    """
    try:
        import httpx
    except ImportError:
        print("[url] httpx 未安装，请: pip install httpx")
        return None
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 (RAG-Ingest)"}) as c:
            r = c.get(url)
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").lower()
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                return load_pdf_bytes(r.content)
            return load_html(r.text, base_url=url)
    except Exception as e:
        print(f"[url] {url}: {e}")
        return None


def load_pdf_bytes(data: bytes) -> str | None:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        print(f"[pdf-bytes] {e}")
        return None


# ---------- 自动分发 ----------

LOADERS_BY_EXT = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html_file,
    ".htm": load_html_file,
}


def load_any(path_or_url: str) -> str | None:
    """统一入口：本地文件按扩展名分发，URL 走在线下载。"""
    if path_or_url.startswith(("http://", "https://")):
        return load_url(path_or_url)
    ext = Path(path_or_url).suffix.lower()
    loader = LOADERS_BY_EXT.get(ext)
    if not loader:
        print(f"[skip] unsupported ext: {ext} ({path_or_url})")
        return None
    return loader(path_or_url)


def collect_files(base_path: str, exts: tuple = (".md", ".pdf", ".docx", ".html", ".txt"),
                  excluded_dirs: tuple = ("node_modules", ".venv", "dist", "build",
                                          "__pycache__", "plugin_daemon", ".git")) -> list[str]:
    """递归收集多种格式文件，按扩展名筛选，排除常见噪声目录。"""
    base = Path(base_path).resolve()
    matched = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        if any(part in excluded_dirs for part in p.parts):
            continue
        matched.append(str(p))
    return matched


if __name__ == "__main__":
    # 简单冒烟测试：尝试加载一个本仓库的 md
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "README.md"
    text = load_any(target)
    if text:
        print(f"OK loaded {len(text)} chars, preview:")
        print(text[:300])
    else:
        print("FAILED")

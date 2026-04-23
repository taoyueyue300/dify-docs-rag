# ?? Git 上传前完整清单

> 你在上传代码前需要完成的所有事项。按顺序执行即可。

---

## 第一阶段：本地验证

### ? 1. 确认 `.env` 和 `.gitignore`

```powershell
# 检查 .env 是否被忽略
git check-ignore -v .env

# 预期输出：.env
# 如果没有输出，说明 .env 可能被提交了，需要清理历史
```

**如果 `.env` 曾被提交过**：

```powershell
# 使用 git-filter-repo 清理历史
pip install git-filter-repo
git filter-repo --path .env --invert-paths

# 然后需要强制推送（仅在私人仓库或得到团队同意时）
git push origin --force-with-lease --all
```

---

### ? 2. 扫描敏感信息

用 ripgrep 扫描潜在泄露的密钥：

```powershell
# 安装 ripgrep（如果还没装）
choco install ripgrep
# 或从 scoop：scoop install ripgrep

# 扫描常见密钥模式
rg -n "(api[_-]?key|sk-|secret|token|password|LLM_API_KEY)" --type py

# 扫描所有文件（包括文本、JSON、配置）
rg -n "(api[_-]?key|sk-|secret)" .
```

**如果发现泄露**：

- 立即轮换 API Key（通知你的 API 提供商）
- 删除泄露内容并 push
- 在 GitHub Issue / Commit 中 **不要提及真实 Key**

---

### ? 3. 检查将被提交的文件

```powershell
# 查看 git 状态
git status

# 确认以下文件 **不会** 被提交：
# ? .env
# ? .venv/
# ? faiss_index/
# ? faiss_index_multi/
# ? __pycache__/
# ? *.pyc
# ? *_report.json（本地测试产物）

# 查看实际会被提交的文件
git add .
git status --short
```

---

### ? 4. 最终扫描：README & 代码

**检查清单**：

- [ ] README.md 中没有真实 API Key（只有 `sk-xxxx` 形式的示例）
- [ ] `.env.example` 是模板形式，不含任何真实信息
- [ ] 代码中没有硬编码 API Key（应该都从 `os.getenv()` 读取）
- [ ] 没有不小心 commit 的日志文件、临时文件

---

## 第二阶段：Git 初始化与提交

### ? 5. 初始化 Git（如果还没做）

```powershell
cd c:\Users\25352\Desktop\实习\dify-docs-rag

# 如果是新仓库
git init

# 如果已经有 remote，检查
git remote -v
```

---

### ? 6. 首次提交

```powershell
# 暂存所有文件（.gitignore 会自动排除 .env 等）
git add .

# 查看最后一遍
git status

# 创建首次提交
git commit -m "feat: init dify-docs-rag with RAG pipeline

- Hybrid retrieval (FAISS vector + BM25 keyword search)
- LLM-based generation with source attribution
- CLI/API/Web UI interfaces
- Comprehensive evaluation & benchmarking suite
- Bilingual README with production-ready security checks"
```

---

### ? 7. 添加远程仓库 & 推送

```powershell
# 假设你已经在 GitHub/GitLab 创建了空仓库

# 添加远程
git remote add origin https://github.com/YOUR_USERNAME/dify-docs-rag.git

# 检查
git remote -v

# 创建并推送主分支
git branch -M main
git push -u origin main

# 验证
git log --oneline -n 5
```

---

## 第三阶段：上传后的维护

### ? 8. 配置 GitHub 分支保护（可选）

在 GitHub Settings → Branches 中：

- 启用 "Require pull request reviews before merging"
- 启用 "Dismiss stale pull request approvals when new commits are pushed"
- 启用 "Require branches to be up to date before merging"

---

### ? 9. 添加更多元数据（可选但推荐）

创建 `LICENSE` 文件（如果还没有）：

```bash
# MIT License 模板
# 或从 GitHub 下载：https://choosealicense.com/licenses/mit/
```

创建 `.github/CONTRIBUTING.md`（贡献指南）：

```markdown
# 贡献指南

感谢你有兴趣贡献代码！

## 开发流程

1. Fork 本仓库
2. 创建 feature branch (`git checkout -b feature/amazing-thing`)
3. 确保 `python benchmark.py` 和 `python eval.py` 通过
4. 提交 PR

## 代码风格

- 使用 4 空格缩进
- PEP 8 命名规范
- 复杂逻辑需要注释
```

---

### ? 10. 打标签（版本管理）

```powershell
# 创建第一个发布标签
git tag -a v0.1.0 -m "Initial release: RAG pipeline with hybrid retrieval"

# 推送标签
git push origin v0.1.0

# 验证
git tag -l
```

---

## 最终安全确认

| 检查项 | 状态 |
|---|---|
| ? `.env` 被正确忽略 | [ ] |
| ? 代码中无硬编码密钥 | [ ] |
| ? `.gitignore` 包含 faiss_index/ | [ ] |
| ? README 中密钥示例用 `sk-xxx` 掩码 | [ ] |
| ? 历史中无泄露的 API Key | [ ] |
| ? 虚拟环境 `.venv/` 被忽略 | [ ] |
| ? 实验产物 `*_report.json` 被忽略 | [ ] |

---

## 常见问题

### Q: 我想修改 README 中的截图占位符

在 README.md 中，找到这一行：

```markdown
![Dify RAG Web UI](https://via.placeholder.com/800x400?text=Streamlit+Web+UI+Demo)
```

替换为：

```markdown
![Dify RAG Web UI](./screenshots/web-ui-demo.png)
```

然后在项目根目录创建 `screenshots/` 文件夹，把你的截图放进去。

### Q: 我想添加更多的中文vs英文支持

所有内容都在 README.md 的 `<details>` 标签中，可以：

1. 中文部分：在 `<summary><h2>?? 中文</h2></summary>` 内
2. 英文部分：在 `<summary><h2>?? English</h2></summary>` 内

修改后 push 即可。

### Q: 上传后发现有密钥泄露怎么办？

**立即操作**：

```powershell
# 1. 轮换 API Key
# （在 LLM 服务商界面操作）

# 2. 通知所有使用该 Key 的地方
# （如果是私人项目，至少通知自己不要用泄露的 Key）

# 3. 清理历史（仅限私人仓库）
git filter-repo --path .env --invert-paths
git push --force-with-lease

# 4. 在 GitHub Settings > Security 中检查密钥泄露警告
```

---

## 推荐的文件结构（完整）

```
dify-docs-rag/
├── .env                          # ?? 本地只有，不提交
├── .env.example                  # ? 提交（模板）
├── .gitignore                    # ? 已更新
├── README.md                     # ? 已重写（中英双语）
├── LICENSE                       # ? MIT License
├── requirements.txt              # ? 依赖清单
├── app.py                        # ? FastAPI
├── chain.py                      # ? RAG链
├── retriever.py                  # ? 混合检索
├── ingest.py                     # ? 文档入库
├── ingest_multi.py               # ? 多源入库
├── loaders.py                    # ? 多格式加载器
├── ui.py                         # ? Streamlit UI
├── eval.py                       # ? 评测
├── run_experiments.py            # ? 实验
├── benchmark.py                  # ? 性能基准
├── .venv/                        # ? 被 .gitignore 忽略
├── faiss_index/                  # ? 被 .gitignore 忽略
├── __pycache__/                  # ? 被 .gitignore 忽略
├── docs/
│   └── 多源知识库扩展指南.md      # ? 提交
├── 代码详解.md                   # ? 提交
├── 使用说明.md                   # ? 提交
├── 性能基准报告.md               # ? 提交
├── 面试问答手册.md               # ? 提交
├── benchmark_report.json         # ? 被 .gitignore 忽略
└── experiment_report.json        # ? 被 .gitignore 忽略
```

---

## 下一步：打磨与推广

上传完成后，可以考虑：

1. **添加 CI/CD**：GitHub Actions 自动运行 `benchmark.py`
2. **写测试**：单元测试 + 集成测试
3. **发布到 PyPI**：让用户 `pip install dify-docs-rag`
4. **完善文档**：补充架构设计、深度学习细节
5. **开源社区**：提交到 Awesome RAG、ProductHunt 等

---

**祝上传顺利！** ?

有任何问题，查看本文档的"常见问题"部分，或提交 GitHub Issue。

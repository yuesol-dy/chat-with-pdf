# 📄 Chat with PDF – 基于本地大模型的文档智能问答

一个完全本地化的 PDF 文档智能问答系统。上传 PDF 文件后，系统自动提取文本、向量化并存入 ChromaDB，随后可通过自然语言提问，由本地大语言模型基于检索增强生成（RAG）给出精准答案。

**全程无需联网，保护数据隐私，纯 CPU 即可在个人笔记本上流畅运行。**

## 🏗️ 系统架构

```
┌──────────────┐      ┌─────────────────┐     ┌──────────────┐
│  Streamlit   │────▶│   LangChain RAG │────▶│  Ollama API  │
│  前端界面     │      │   检索增强生成   │     │  qwen2:7b    │
└──────────────┘      └────────┬────────┘     └──────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    ChromaDB       │
                    │  向量数据库（本地） │
                    └───────────────────┘
```

## 🛠️ 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 界面 | [Streamlit](https://streamlit.io) | 纯 Python Web UI |
| RAG 框架 | [LangChain](https://langchain.com) | 检索增强生成管道 |
| LLM | [Ollama](https://ollama.com) + Qwen2 7B | 本地推理，无需 GPU |
| 嵌入模型 | nomic-embed-text (137MB) | 轻量文本向量化 |
| 向量数据库 | [ChromaDB](https://trychroma.com) | 本地持久化存储 |
| PDF 解析 | [PyMuPDF](https://pymupdf.readthedocs.io) | 高性能 PDF 文本提取 |

## 📋 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/download)（已安装并在后台运行）
- 约 8GB 以上内存（模型运行时占用约 4-5GB）
- 无 GPU 要求（纯 CPU 推理）

## 🚀 快速开始

### 1. 安装 Ollama 并拉取模型

```bash
# 安装 Ollama（https://ollama.com/download）
ollama pull qwen2:7b        # 大语言模型（4.4GB）
ollama pull nomic-embed-text # 嵌入模型（137MB）
```

### 2. 克隆项目

```bash
git clone git@github.com:yuesol-dy/chat-with-pdf.git
cd chat-with-pdf
```

### 3. 创建虚拟环境并安装依赖

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 4. 启动 Ollama 服务

在新终端窗口启动（需启用 embeddings 支持）：

```bash
# Windows (Cmd)
set OLLAMA_EMBEDDINGS=1 && ollama serve

# macOS / Linux
OLLAMA_EMBEDDINGS=1 ollama serve
```

### 5. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`，上传 PDF → 点击"处理文件" → 开始提问！

## 📁 项目结构

```
chat-with-pdf/
├── app.py              # Streamlit 主界面
├── rag_pipeline.py     # RAG 核心逻辑（向量化、检索、生成）
├── utils.py            # PDF 文本提取、分块工具
├── requirements.txt    # Python 依赖
├── .gitignore
└── README.md
```

## 💡 核心功能

### PDF 文本提取（utils.py）

- **`extract_text_from_pdf(pdf_path)`** — 使用 PyMuPDF 提取 PDF 全部文本
- **`split_text_into_chunks(text)`** — 使用递归字符分割器按语义分块（500 字符/块，50 字符重叠）

### RAG 管道（rag_pipeline.py）

- **自定义 `OllamaEmbeddingsHttp`** — 通过 HTTP API 直接调用 Ollama embedding，批量发送文本块
- **`create_vector_store(texts)`** — 创建 ChromaDB 向量存储并持久化到本地
- **`load_vector_store()`** — 加载已有向量数据库（避免重复处理）
- **`build_qa_chain(vectorstore)`** — 构造 LangChain RetrievalQA 链，基于上下文回答问题

### 交互界面（app.py）

- 侧边栏上传 PDF（支持多文件）
- 一键处理：提取 → 分块 → 向量化 → 存储
- 对话式问答界面，支持多轮对话
- 自动加载已有向量数据库

## ⚡ 性能说明

本系统运行在**纯 CPU** 环境：

- **qwen2:7b** 推理速度约 **4-5 tokens/s**
- 单个问题回答耗时约 **10-30 秒**（取决于上下文长度）
- **1.8MB PDF** 首次处理约 40-60 秒（embedding + 向量存储）
- 后续加载向量库瞬间完成

> 如需更快的响应速度，可更换更小的模型如 `qwen2:1.5b`。

## 🔧 配置说明

| 配置项 | 位置 | 说明 |
|--------|------|------|
| chunk_size | `utils.py` | 文本分块大小，默认 500 |
| chunk_overlap | `utils.py` | 分块重叠量，默认 50 |
| 检索数量 k | `rag_pipeline.py` | 检索最相关的 k 个文本块，默认 3 |
| LLM temperature | `rag_pipeline.py` | 模型温度，默认 0.1（更精确） |
| 嵌入模型 | `rag_pipeline.py` | 默认 nomic-embed-text |
| LLM 模型 | `rag_pipeline.py` | 默认 qwen2:7b |

## 🔄 更换模型

修改 `rag_pipeline.py` 中的模型名称即可：

```python
# 更换 LLM
llm = OllamaLLM(model="llama3.2", temperature=0.1)

# 更换嵌入模型
embeddings = OllamaEmbeddingsHttp(model="mxbai-embed-large")
```

确保先 `ollama pull <model-name>` 拉取对应模型。

## ❓ 常见问题

### Q: 报错 `503 / 502` embedding 错误
A: Ollama 服务未启用 embeddings。使用 `OLLAMA_EMBEDDINGS=1 ollama serve` 启动。

### Q: "思考中" 卡住很久
A: qwen2:7b 在纯 CPU 上推理较慢，属于正常现象。等待片刻即可。可换用更小模型提速。

### Q: ChromaDB 报错
A: 删除 `chroma_db/` 目录后重新处理 PDF，或检查磁盘空间。

### Q: 提取不到文字
A: PDF 可能是扫描版（图片）而非文本版。本项目暂不支持 OCR，需要 PDF 包含可选文字层。

## 📄 许可

MIT License

---

**Made by [yuesol-dy](https://github.com/yuesol-dy)**










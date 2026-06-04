import requests
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.embeddings import Embeddings


class OllamaEmbeddingsHttp(Embeddings):
    """用 requests 直接调用 Ollama embedding API，绕过 ollama 客户端兼容问题"""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 批量发送：Ollama API 支持 input 为字符串列表
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=600
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_vector_store(texts: list[str], persist_dir: str = "./chroma_db"):
    """从文本块列表创建 Chroma 向量存储（持久化到磁盘）"""
    embeddings = OllamaEmbeddingsHttp(model="nomic-embed-text")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vectorstore


def load_vector_store(persist_dir: str = "./chroma_db"):
    """加载已有的本地向量存储"""
    embeddings = OllamaEmbeddingsHttp(model="nomic-embed-text")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def build_qa_chain(vectorstore):
    """构建基于检索增强的问答链"""
    llm = OllamaLLM(model="qwen2:7b", temperature=0.1)
    prompt_template = """你是一个有帮助的AI助手。请根据以下上下文回答用户的问题。
如果你不知道答案，就直接说不知道，不要编造。

上下文：
{context}

问题：{question}

有帮助的回答："""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa
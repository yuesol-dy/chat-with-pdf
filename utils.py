import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 文件路径提取所有页面文本"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50):
    """将长文本按语义切分为适合嵌入的块"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    return splitter.split_text(text)
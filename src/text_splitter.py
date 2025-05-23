"""
文本分块模块，用于将长文本切分成适合向量化的小段落
"""
import logging
from typing import List, Dict, Any, Optional, Union
import nltk
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter

from utils.helpers import time_function
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保nltk资源已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextSplitter:
    """文本分块器类，提供多种文本分块策略，默认使用递归字符分割"""
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE, 
        chunk_overlap: int = CHUNK_OVERLAP,
        splitter_type: str = "recursive"
    ):
        """
        初始化文本分块器
        
        Args:
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            splitter_type: 分块器类型，可选值：recursive, character, token
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type
        
        # 针对中文和英文的分隔符优化
        if splitter_type == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    # 首先尝试按段落分割
                    "\n\n", "\n", 
                    # 然后按句子分割
                    "。", "！", "？", "!", "?", ".", 
                    # 然后按分句符号分割
                    "；", ";", "，", ",", 
                    # 最后按词或字符分割
                    " ", ""
                ],
                keep_separator=False
            )
            logger.info("使用递归字符文本分割器，针对中英文混合内容进行了优化")
        elif splitter_type == "character":
            self.text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
        elif splitter_type == "token":
            self.text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            logger.warning(f"不支持的分块器类型: {splitter_type}，使用默认的RecursiveCharacterTextSplitter")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
            
        logger.info(f"初始化分块器: {splitter_type}, 块大小: {chunk_size}, 重叠大小: {chunk_overlap}")
    
    @time_function
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档
        
        Args:
            documents: 文档列表
            
        Returns:
            分割后的文档列表
        """
        if not documents:
            logger.warning("没有文档可分割")
            return []
            
        logger.info(f"分割 {len(documents)} 个文档")
        try:
            if self.splitter_type == "recursive":
                # 对于递归分割器，保留文档元数据并确保完整性
                split_docs = []
                for doc in documents:
                    chunks = self.text_splitter.split_documents([doc])
                    # 确保每个分块都包含源文档信息
                    for chunk in chunks:
                        if "source" not in chunk.metadata and "source" in doc.metadata:
                            chunk.metadata["source"] = doc.metadata["source"]
                    split_docs.extend(chunks)
            else:
                split_docs = self.text_splitter.split_documents(documents)
                
            logger.info(f"分割完成，共 {len(split_docs)} 个文本块")
            return split_docs
        except Exception as e:
            logger.error(f"分割文档失败: {str(e)}")
            return documents
    
    @time_function
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        分割单个文本
        
        Args:
            text: 文本内容
            metadata: 元数据
            
        Returns:
            分割后的文档列表
        """
        if not text:
            logger.warning("没有文本可分割")
            return []
            
        metadata = metadata or {}
        logger.info(f"分割文本，长度: {len(text)}")
        
        try:
            # 对于递归分割器，我们可以设置不同的分割策略
            split_texts = self.text_splitter.split_text(text)
            documents = [Document(page_content=t, metadata=metadata.copy()) for t in split_texts]
            logger.info(f"分割完成，共 {len(documents)} 个文本块")
            return documents
        except Exception as e:
            logger.error(f"分割文本失败: {str(e)}")
            return [Document(page_content=text, metadata=metadata)]
            
def get_text_splitter(
    splitter_type: str = "recursive",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> TextSplitter:
    """
    获取文本分块器实例
    
    Args:
        splitter_type: 分块器类型，默认使用recursive
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        
    Returns:
        TextSplitter实例
    """
    return TextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_type=splitter_type
    ) 
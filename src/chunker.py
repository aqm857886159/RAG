"""
文本分块模块，用于将文本分割成适合向量化的片段
"""
import re
import logging
from typing import List, Dict, Any, Optional, Union
from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

from config.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Chunker:
    """文本分块器类，用于文本分割"""
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        splitter_type: str = "recursive"
    ):
        """
        初始化分块器
        
        Args:
            chunk_size: 块大小(字符数)
            chunk_overlap: 块重叠大小(字符数)
            splitter_type: 分割器类型，支持 recursive, character, token
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type.lower()
        
        # 初始化分割器
        if self.splitter_type == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            logger.info(f"初始化递归分块器: 块大小={chunk_size}, 重叠大小={chunk_overlap}")
        elif self.splitter_type == "character":
            self.text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator="\n"
            )
            logger.info(f"初始化字符分块器: 块大小={chunk_size}, 重叠大小={chunk_overlap}")
        elif self.splitter_type == "token":
            self.text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info(f"初始化Token分块器: 块大小={chunk_size}, 重叠大小={chunk_overlap}")
        else:
            logger.error(f"不支持的分割器类型: {splitter_type}")
            raise ValueError(f"不支持的分割器类型: {splitter_type}")
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本
        
        Args:
            text: 输入文本
            
        Returns:
            分割后的文本列表
        """
        if not text:
            logger.warning("输入文本为空，无法分割")
            return []
            
        logger.info(f"开始分割文本，长度: {len(text)}")
        
        # 预处理：删除多余空白行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 使用文本分割器进行分割
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"文本分割完成，共 {len(chunks)} 个块")
            return chunks
        except Exception as e:
            logger.error(f"文本分割失败: {str(e)}")
            # 如果分割失败，尝试简单分割
            return self._fallback_split(text)
    
    def _fallback_split(self, text: str) -> List[str]:
        """
        备用的简单分割方法
        
        Args:
            text: 输入文本
            
        Returns:
            分割后的文本列表
        """
        logger.info("使用备用分割方法")
        
        # 按段落分割
        paragraphs = text.split("\n\n")
        
        # 合并段落成块
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"备用分割完成，共 {len(chunks)} 个块")
        return chunks
    
    def create_documents(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        从文本创建文档
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            Document对象列表
        """
        if not text:
            logger.warning("输入文本为空，无法创建文档")
            return []
            
        # 设置默认元数据
        if metadata is None:
            metadata = {}
        
        # 分割文本
        chunks = self.split_text(text)
        
        # 创建文档
        documents = []
        for i, chunk in enumerate(chunks):
            # 为每个文档添加索引信息
            doc_metadata = metadata.copy()
            doc_metadata["chunk_index"] = i
            doc_metadata["chunk_count"] = len(chunks)
            
            # 创建Document对象
            document = Document(
                page_content=chunk,
                metadata=doc_metadata
            )
            documents.append(document)
        
        logger.info(f"文档创建完成，共 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表
        
        Args:
            documents: 输入文档列表
            
        Returns:
            分割后的文档列表
        """
        if not documents:
            logger.warning("输入文档为空，无法分割")
            return []
            
        logger.info(f"开始分割文档，共 {len(documents)} 个文档")
        
        try:
            # 使用文本分割器分割文档
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档分割完成，分割后共 {len(split_docs)} 个文档")
            return split_docs
        except Exception as e:
            logger.error(f"文档分割失败: {str(e)}")
            
            # 如果分割失败，手动分割
            split_docs = []
            for doc in documents:
                # 获取原始元数据
                metadata = doc.metadata.copy()
                
                # 分割文本
                chunks = self._fallback_split(doc.page_content)
                
                # 创建新文档
                for i, chunk in enumerate(chunks):
                    # 添加分块信息
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["chunk_count"] = len(chunks)
                    
                    # 创建Document对象
                    split_doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    split_docs.append(split_doc)
            
            logger.info(f"备用文档分割完成，分割后共 {len(split_docs)} 个文档")
            return split_docs

def get_chunker(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    splitter_type: str = "recursive"
) -> Chunker:
    """
    获取分块器实例
    
    Args:
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
        splitter_type: 分割器类型
        
    Returns:
        Chunker实例
    """
    return Chunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_type=splitter_type
    ) 
"""
自定义检索器模块，避免使用langchain.retrievers导入
"""
import logging
from typing import List, Dict, Any, Optional, Callable

from langchain.docstore.document import Document
from langchain.schema.language_model import BaseLanguageModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContextualCompressionRetriever:
    """自定义上下文压缩检索器"""
    
    def __init__(
        self,
        base_retriever,
        document_compressor,
        verbose: bool = False
    ):
        """
        初始化上下文压缩检索器
        
        Args:
            base_retriever: 基础检索器，提供初步的文档检索
            document_compressor: 文档压缩器，对检索结果进行压缩
            verbose: 是否输出详细日志
        """
        self.base_retriever = base_retriever
        self.document_compressor = document_compressor
        self.verbose = verbose
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取相关文档并进行压缩
        
        Args:
            query: 查询文本
            
        Returns:
            压缩后的相关文档列表
        """
        # 使用基础检索器获取初步结果
        docs = self.base_retriever.get_relevant_documents(query)
        if self.verbose:
            logger.info(f"基础检索器返回了 {len(docs)} 个文档")
            
        # 使用文档压缩器进行压缩
        compressed_docs = self.document_compressor.compress_documents(docs, query)
        if self.verbose:
            logger.info(f"压缩后剩余 {len(compressed_docs)} 个文档")
            
        return compressed_docs

class LLMChainExtractor:
    """使用LLM从文档中提取相关上下文"""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        verbose: bool = False
    ):
        """
        初始化LLM链式提取器
        
        Args:
            llm: 语言模型
            verbose: 是否输出详细日志
        """
        self.llm = llm
        self.verbose = verbose
        
    def compress_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        从文档中提取与查询相关的内容
        
        Args:
            documents: 文档列表
            query: 查询文本
            
        Returns:
            压缩后的文档列表
        """
        if not documents:
            return []
            
        compressed_docs = []
        for doc in documents:
            try:
                # 构建提示
                prompt = f"""
                从以下文本中提取与问题相关的信息:
                
                问题: {query}
                
                文本:
                {doc.page_content}
                
                仅提取与问题直接相关的信息。如果没有相关信息，返回"没有相关信息"。
                """
                
                # 调用LLM
                extracted_content = self.llm.predict(prompt)
                
                # 如果有相关内容，创建新的文档
                if extracted_content and "没有相关信息" not in extracted_content:
                    compressed_docs.append(
                        Document(
                            page_content=extracted_content,
                            metadata=doc.metadata
                        )
                    )
            except Exception as e:
                if self.verbose:
                    logger.error(f"压缩文档时出错: {str(e)}")
                # 在出错时保留原始文档
                compressed_docs.append(doc)
                
        return compressed_docs 
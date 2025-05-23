"""
生成模块，用于LLM生成功能和RAG检索增强
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, Tuple

from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain_core.memory import BaseMemory
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import _get_chat_history

from src.llm import get_llm
from src.vectorizer import Vectorizer
from config.config import get_model_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """生成器基类"""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[BaseMemory] = None,
        **kwargs
    ):
        """
        初始化生成器基类
        
        Args:
            llm: 语言模型
            memory: 对话记忆
        """
        self.llm = llm or get_llm()
        
        # 初始化记忆
        self.memory = memory or ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
        
        logger.info(f"初始化生成器基类: LLM={type(self.llm).__name__}")
    
    @abstractmethod
    def generate(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        生成回答的抽象方法
        
        Args:
            query: 用户查询
            
        Returns:
            包含回答的字典
        """
        pass
    
    def reset_memory(self) -> None:
        """重置对话记忆"""
        if hasattr(self.memory, "clear"):
            self.memory.clear()
        elif hasattr(self.memory, "chat_memory") and hasattr(self.memory.chat_memory, "clear"):
            self.memory.chat_memory.clear()
            
        logger.info("已重置对话记忆")


class SimpleLLMGenerator(BaseGenerator):
    """简单LLM生成器，不使用检索功能"""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[BaseMemory] = None,
        **kwargs
    ):
        """
        初始化简单LLM生成器
        
        Args:
            llm: 语言模型
            memory: 对话记忆
        """
        super().__init__(llm=llm, memory=memory, **kwargs)
        
        # 初始化记忆
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
            )
        
        logger.info(f"初始化简单LLM生成器")
            
    def generate(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        生成回答
        
        Args:
            query: 用户查询
            
        Returns:
            包含回答的字典
        """
        try:
            # 直接使用LLM进行对话
            result = self.llm.predict_messages(
                messages=[{"role": "user", "content": query}],
                callbacks=kwargs.get("callbacks", None)
            )
                    
            # 提取回答文本
            answer = result.content
            
            # 使用memory.save_context更新对话历史
            self.memory.save_context({"input": query}, {"output": answer})
            
            logger.info(f"生成回答: 长度={len(answer)}")
            
            return {
                "answer": answer,
                "source_documents": [],
                "is_rag_mode": False
            }
            
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return {
                "answer": f"生成回答时出错: {str(e)}。请稍后再试。",
                "source_documents": [],
                "is_rag_mode": False,
                "error": str(e)
            }


class RAGGenerator(BaseGenerator):
    """RAG检索增强生成器"""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        memory: Optional[BaseMemory] = None,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[BaseRetriever] = None,
        **kwargs
    ):
        """
        初始化RAG检索增强生成器
        
        Args:
            llm: 语言模型
            memory: 对话记忆
            vector_store: 向量存储
            retriever: 检索器
        """
        super().__init__(llm=llm, memory=memory, **kwargs)
        
        self.vector_store = vector_store
        
        # 初始化检索器
        self.retriever = None
        
        # 设置检索器
        if retriever:
            self.retriever = retriever
            logger.info(f"使用提供的检索器: {type(retriever).__name__}")
        elif vector_store:
            try:
                # 尝试使用向量存储的as_retriever方法
                self.retriever = vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                logger.info(f"已创建基础检索器")
            except Exception as e:
                logger.error(f"创建检索器失败: {str(e)}")
                self.retriever = None
        
        # 创建RAG链
        self.chain = self._create_chain()
        logger.info(f"初始化RAG生成器: 检索器状态={self.retriever is not None}")
    
    def _create_chain(self) -> Optional[ConversationalRetrievalChain]:
        """
        创建对话检索链
        
        Returns:
            对话检索链
        """
        if not self.retriever:
            logger.warning("没有检索器，无法创建对话检索链")
            return None
        
        try:
            # 创建对话检索链
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                get_chat_history=_get_chat_history
            )
            
            return chain
            
        except Exception as e:
            logger.error(f"创建对话检索链失败: {str(e)}")
            return None
    
    def generate(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        生成回答
        
        Args:
            query: 用户查询
            
        Returns:
            包含回答的字典
        """
        if not self.retriever or not self.chain:
            logger.warning("没有检索器或对话链，无法使用RAG模式，回退到简单LLM模式")
            # 回退到简单LLM模式
            simple_generator = SimpleLLMGenerator(llm=self.llm, memory=self.memory)
            result = simple_generator.generate(query, **kwargs)
            # 添加提示信息
            result["answer"] = (
                "【提示：当前处于纯LLM模式，没有使用知识库。请先上传并索引文档。】\n\n" + 
                result["answer"]
            )
            return result
        
        try:
            # 使用对话检索链生成回答
            result = self.chain.invoke(
                {"question": query},
                callbacks=kwargs.get("callbacks", None)
            )
            
            # 提取回答和源文档
            answer = result.get("answer", "未能生成回答")
            source_documents = result.get("source_documents", [])
            
            logger.info(f"生成RAG回答: 长度={len(answer)}, 源文档数={len(source_documents)}")
            
            return {
                "answer": answer,
                "source_documents": source_documents,
                "is_rag_mode": True
            }
            
        except Exception as e:
            logger.error(f"生成RAG回答失败: {str(e)}")
            
            # 尝试回退到简单LLM模式
            try:
                logger.info("尝试回退到简单LLM模式")
                simple_generator = SimpleLLMGenerator(llm=self.llm, memory=self.memory)
                result = simple_generator.generate(query, **kwargs)
                # 添加错误提示
                result["answer"] = (
                    f"【RAG模式出错: {str(e)}。已回退到纯LLM模式。】\n\n" + 
                    result["answer"]
                )
                return result
            except Exception as inner_e:
                logger.error(f"回退到简单LLM模式也失败: {str(inner_e)}")
                return {
                    "answer": f"生成回答时出错: {str(e)}，回退也失败: {str(inner_e)}。请稍后再试。",
                    "source_documents": [],
                    "is_rag_mode": False,
                    "error": str(e)
                }

        
def get_generator(
    use_rag: bool = True,
    llm: Optional[BaseChatModel] = None,
    vector_store: Optional[VectorStore] = None,
    retriever: Optional[BaseRetriever] = None,
    memory: Optional[BaseMemory] = None,
    **kwargs
) -> Union[RAGGenerator, SimpleLLMGenerator]:
    """
    工厂函数，获取生成器实例
    
    Args:
        use_rag: 是否使用RAG
        llm: 语言模型
        vector_store: 向量存储
        retriever: 检索器
        memory: 对话记忆
        
    Returns:
        生成器实例
    """
    # 确保有LLM
    if not llm:
        try:
            llm = get_llm(**kwargs)
        except Exception as e:
            logger.error(f"获取LLM失败: {str(e)}")
            # 尝试使用备选模型配置
            try:
                model_config = get_model_config()
                fallback_model = model_config.get("fallback_model", "gpt-3.5-turbo")
                logger.info(f"尝试使用备选模型: {fallback_model}")
                llm = get_llm(model_name=fallback_model)
            except Exception as inner_e:
                logger.error(f"获取备选LLM也失败: {str(inner_e)}")
                raise ValueError(f"无法初始化LLM: {str(e)}, 备选也失败: {str(inner_e)}")
    
    # 初始化记忆
    if not memory:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    # 根据需求创建不同类型的生成器
    if use_rag and (vector_store or retriever):
        try:
            # 创建RAG生成器
            return RAGGenerator(
                llm=llm,
                memory=memory,
                vector_store=vector_store,
        retriever=retriever,
                **kwargs
            )
        except Exception as e:
            logger.error(f"创建RAG生成器失败: {str(e)}，回退到简单LLM生成器")
            # 在出错时回退到简单LLM生成器
            return SimpleLLMGenerator(
                llm=llm,
                memory=memory,
                **kwargs
            )
    else:
        # 创建简单LLM生成器
        return SimpleLLMGenerator(
            llm=llm,
            memory=memory,
            **kwargs
    ) 
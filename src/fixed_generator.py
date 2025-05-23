"""
修复版生成模块，用于LLM生成功能和RAG检索增强
"""
import logging
import os
from typing import List, Dict, Optional, Any, Union, Tuple

from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import HumanMessage
from langchain_core.memory import BaseMemory
from langchain_core.language_models import BaseChatModel
from langchain.memory import ConversationBufferMemory

from src.llm import get_llm
from config.config import get_model_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BaseGenerator:
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
    
    def generate(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        生成回答的抽象方法
        
        Args:
            query: 用户查询
            
        Returns:
            包含回答的字典
        """
        raise NotImplementedError("子类必须实现generate方法")
    
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
            # 使用新的invoke方法而不是predict_messages
            message = HumanMessage(content=query)
            result = self.llm.invoke([message], callbacks=kwargs.get("callbacks", None))
                    
            # 提取回答文本
            answer = result.content if hasattr(result, 'content') else str(result)
            
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
    """RAG检索增强生成器 - 使用自定义RAG逻辑，不依赖LangChain的ConversationalRetrievalChain"""
    
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
        self.retriever = retriever
        
        # 如果没有提供检索器但提供了向量存储，尝试创建检索器
        if not retriever and vector_store:
            try:
                self.retriever = vector_store.as_retriever(
                    search_kwargs={"k": 5}
                )
                logger.info(f"已创建基础检索器")
            except Exception as e:
                logger.error(f"创建检索器失败: {str(e)}")
                self.retriever = None
        
        logger.info(f"初始化RAG生成器: 检索器状态={self.retriever is not None}")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        if not self.retriever:
            return []
            
        try:
            # 使用检索器获取相关文档
            docs = self.retriever.get_relevant_documents(query)
            logger.info(f"检索到 {len(docs)} 个相关文档")
            return docs
        except Exception as e:
            logger.error(f"检索相关文档失败: {str(e)}")
            return []
    
    def _format_chat_history(self) -> str:
        """
        格式化聊天历史
        
        Returns:
            格式化后的聊天历史文本
        """
        if not hasattr(self.memory, "chat_memory") or not self.memory.chat_memory.messages:
            return ""
            
        # 格式化聊天历史
        formatted_history = []
        for message in self.memory.chat_memory.messages:
            if hasattr(message, "type") and hasattr(message, "content"):
                prefix = "用户: " if message.type == "human" else "AI: "
                formatted_history.append(f"{prefix}{message.content}")
                
        return "\n".join(formatted_history)
    
    def _create_rag_prompt(self, query: str, documents: List[Document]) -> str:
        """
        创建RAG提示
        
        Args:
            query: 查询文本
            documents: 相关文档
            
        Returns:
            完整的RAG提示
        """
        # 提取文档内容
        context = "\n\n".join([f"文档 {i+1}:\n{doc.page_content}" for i, doc in enumerate(documents)])
        
        # 获取聊天历史
        chat_history = self._format_chat_history()
        
        # 创建提示
        prompt = f"""请基于以下文档回答问题。只使用提供的文档中的信息。如果文档中没有相关信息，请说明你不知道答案。

{context}

"""
        
        # 添加聊天历史（如果有）
        if chat_history:
            prompt += f"""

以下是之前的对话历史:
{chat_history}

"""
        
        # 添加用户问题
        prompt += f"""
用户问题: {query}

回答:
"""
        
        return prompt
        
    def generate(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        生成RAG回答
        
        Args:
            query: 用户查询
            
        Returns:
            包含回答和源文档的字典
        """
        # 检查是否有检索器
        if not self.retriever:
            logger.warning("没有检索器或对话链，无法使用RAG模式，回退到简单LLM模式")
            # 回退到简单LLM模式
            simple_generator = SimpleLLMGenerator(llm=self.llm)
            return simple_generator.generate(query, **kwargs)
        
        try:
            # 获取相关文档
            docs = self._get_relevant_documents(query)
            
            if not docs:
                logger.warning("没有找到相关文档，回退到简单LLM模式")
                # 回退到简单LLM模式
                simple_generator = SimpleLLMGenerator(llm=self.llm)
                return simple_generator.generate(query, **kwargs)
            
            # 创建RAG提示
            prompt = self._create_rag_prompt(query, docs)
            
            # 调用LLM
            message = HumanMessage(content=prompt)
            result = self.llm.invoke([message], callbacks=kwargs.get("callbacks", None))
            
            # 提取回答
            answer = result.content if hasattr(result, 'content') else str(result)
            
            # 保存到对话历史
            self.memory.save_context({"input": query}, {"output": answer})
            
            logger.info(f"生成RAG回答: 长度={len(answer)}, 源文档数={len(docs)}")
            
            return {
                "answer": answer,
                "source_documents": docs,
                "is_rag_mode": True
            }
            
        except Exception as e:
            logger.error(f"生成RAG回答失败: {str(e)}")
            # 回退到简单LLM模式
            try:
                simple_generator = SimpleLLMGenerator(llm=self.llm)
                result = simple_generator.generate(query, **kwargs)
                result["error"] = str(e)
                return result
            except:
                return {
                    "answer": f"生成回答时出错: {str(e)}。请稍后再试。",
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
    provider: str = None,
    **kwargs
) -> Union[RAGGenerator, SimpleLLMGenerator]:
    """
    获取生成器实例
    
    Args:
        use_rag: 是否使用RAG
        llm: 语言模型
        vector_store: 向量存储
        retriever: 检索器
        memory: 对话记忆
        provider: 模型提供商
        
    Returns:
        生成器实例
    """
    # 如果指定了provider，获取对应的LLM
    if provider and not llm:
        llm = get_llm(provider=provider)
    else:
        llm = llm or get_llm()
    
    if use_rag and (vector_store or retriever):
        return RAGGenerator(
            llm=llm,
            memory=memory,
            vector_store=vector_store,
            retriever=retriever,
            **kwargs
        )
    else:
        return SimpleLLMGenerator(
            llm=llm,
            memory=memory,
            **kwargs
        ) 
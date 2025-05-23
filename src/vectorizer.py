"""
向量化模块，用于将文本转换为向量并建立索引
"""
import os
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch

from utils.helpers import time_function, ensure_directory
from config.config import EMBEDDING_MODEL, VECTOR_STORE_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleMemoryVectorStore:
    """简单的内存向量存储，用于在无法创建其他向量存储时作为后备选项"""
    
    def __init__(self, embeddings):
        """
        初始化简单内存向量存储
        
        Args:
            embeddings: 嵌入函数
        """
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
        logger.info("初始化简单内存向量存储")
        
    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表
        """
        if not documents:
            return
            
        logger.info(f"向简单内存向量存储添加 {len(documents)} 个文档")
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.vectors.append(embeddings[i])
            
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if not self.documents:
            logger.warning("向量存储为空，无法进行搜索")
            return []
            
        logger.info(f"在简单内存向量存储中搜索: {query}, k={k}")
        
        # 计算查询向量
        query_embedding = self.embeddings.embed_query(query)
        
        # 计算相似度
        similarities = []
        for vector in self.vectors:
            similarity = self._cosine_similarity(query_embedding, vector)
            similarities.append(similarity)
            
        # 排序并返回前k个结果
        sorted_indices = np.argsort(similarities)[::-1][:k]
        return [self.documents[i] for i in sorted_indices]
        
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度
        """
        norm1 = np.sqrt(sum(x*x for x in vec1))
        norm2 = np.sqrt(sum(x*x for x in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        dot_product = sum(x*y for x, y in zip(vec1, vec2))
        return dot_product / (norm1 * norm2)
        
    def as_retriever(self, **kwargs):
        """
        将向量存储转换为检索器
        
        Returns:
            检索器
        """
        from langchain.retrievers import VectorStoreRetriever
        search_kwargs = kwargs.get("search_kwargs", {})
        search_kwargs["k"] = search_kwargs.get("k", 5)
        
        return VectorStoreRetriever(
            vectorstore=self,
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        
    def get_document_count(self) -> int:
        """
        获取文档数量
        
        Returns:
            文档数量
        """
        return len(self.documents)
    
class Vectorizer:
    """向量化器类，用于文本向量化和索引管理"""
    
    def __init__(
        self, 
        embedding_model_name: str = EMBEDDING_MODEL,
        vector_store_dir: str = str(VECTOR_STORE_DIR),
        vector_store_type: str = "faiss"
    ):
        """
        初始化向量化器
        
        Args:
            embedding_model_name: 嵌入模型名称
            vector_store_dir: 向量存储目录
            vector_store_type: 向量存储类型，支持 faiss 和 chroma
        """
        self.embedding_model_name = embedding_model_name
        self.vector_store_dir = vector_store_dir
        self.vector_store_type = vector_store_type.lower()
        
        # 确保向量存储目录存在
        ensure_directory(vector_store_dir)
        
        # 初始化嵌入模型
        try:
            # 检查是否使用的是BGE模型
            if "bge" in embedding_model_name.lower():
                # BGE模型的特殊配置
                model_kwargs = {"device": "cpu"}
                encode_kwargs = {
                    "normalize_embeddings": True,  # BGE模型推荐进行归一化
                    "batch_size": 8  # 处理较大文档时避免OOM
                }
                logger.info(f"使用BGE模型特定配置初始化: {embedding_model_name}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
            else:
                # 普通模型初始化
                self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
                
            logger.info(f"初始化嵌入模型: {embedding_model_name}")
        except Exception as e:
            logger.error(f"初始化嵌入模型失败: {str(e)}")
            raise
    
    @time_function
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本的嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            logger.warning("没有文本可向量化")
            return []
            
        logger.info(f"向量化 {len(texts)} 个文本")
        try:
            # 对于BGE模型，添加特殊处理
            if "bge" in self.embedding_model_name.lower():
                # BGE模型推荐在查询前添加前缀以提高效果
                processed_texts = []
                for text in texts:
                    # 判断是否为查询文本（通常较短）
                    if len(text) < 200:  # 假设短于200字符的是查询
                        processed_texts.append(f"查询: {text}")
                    else:
                        # 较长文本作为文档，不添加前缀
                        processed_texts.append(text)
                embeddings = self.embeddings.embed_documents(processed_texts)
            else:
                embeddings = self.embeddings.embed_documents(texts)
                
            logger.info(f"向量化完成，维度: {len(embeddings[0])}")
            return embeddings
        except Exception as e:
            logger.error(f"向量化文本失败: {str(e)}")
            return []
    
    def _get_safe_index_path(self, index_name: str) -> str:
        """
        获取安全的索引路径，处理中文或特殊字符路径问题
        
        Args:
            index_name: 索引名称
            
        Returns:
            安全的索引路径
        """
        # 原始路径
        original_path = os.path.join(self.vector_store_dir, index_name)
        
        # 检查路径是否包含非ASCII字符
        has_non_ascii = any(ord(c) > 127 for c in original_path)
        
        if not has_non_ascii:
            # 尝试验证路径可用性
            try:
                ensure_directory(original_path)
                # 测试写入权限
                test_file = os.path.join(original_path, ".test_write")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                logger.info(f"原始路径可用: {original_path}")
                return original_path
            except Exception as e:
                logger.warning(f"原始路径不可用: {original_path}, 错误: {str(e)}")
                # 继续尝试备用路径
        else:
            logger.warning(f"路径包含非ASCII字符，可能导致保存问题: {original_path}")
        
        # 备用路径1: 用户主目录
        try:
            backup_dir = os.path.join(os.path.expanduser("~"), "rag_vector_store", index_name)
            logger.info(f"尝试用户主目录备用路径: {backup_dir}")
            ensure_directory(backup_dir)
            # 测试写入权限
            test_file = os.path.join(backup_dir, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return backup_dir
        except Exception as e:
            logger.warning(f"用户主目录备用路径不可用: {str(e)}")
        
        # 备用路径2: 临时目录
        try:
            temp_dir = os.path.join(tempfile.gettempdir(), "rag_vector_store", index_name)
            logger.info(f"尝试临时目录备用路径: {temp_dir}")
            ensure_directory(temp_dir)
            return temp_dir
        except Exception as e:
            logger.warning(f"临时目录备用路径不可用: {str(e)}")
            
        # 最后尝试使用当前目录
        try:
            current_dir = os.path.join(os.getcwd(), "rag_vector_store", index_name)
            logger.info(f"尝试当前目录备用路径: {current_dir}")
            ensure_directory(current_dir)
            return current_dir
        except Exception as e:
            logger.error(f"所有备用路径都不可用: {str(e)}")
            # 返回原始路径，但可能会失败
            return original_path
    
    @time_function
    def create_vector_store(
        self, 
        documents: List[Document],
        index_name: str = "default"
    ) -> Union[FAISS, Chroma, DocArrayInMemorySearch, None]:
        """
        从文档创建向量存储
        
        Args:
            documents: 文档列表
            index_name: 索引名称
            
        Returns:
            向量存储对象
        """
        if not documents:
            logger.warning("没有文档可索引")
            return None
            
        logger.info(f"创建向量存储: {self.vector_store_type}, 索引名称: {index_name}")
        
        # 获取安全的索引路径
        index_path = self._get_safe_index_path(index_name)
        
        try:
            if self.vector_store_type == "faiss":
                # 使用内存中创建，然后再保存到磁盘
                vector_store = FAISS.from_documents(documents, self.embeddings)
                
                try:
                    vector_store.save_local(index_path)
                    logger.info(f"FAISS向量存储已创建并保存到: {index_path}")
                except Exception as save_error:
                    logger.error(f"FAISS保存失败: {str(save_error)}")
                    logger.warning("无法保存到磁盘，返回内存中的向量存储")
                
            elif self.vector_store_type == "chroma":
                # 对于Chroma，直接使用persist_directory参数
                try:
                    vector_store = Chroma.from_documents(
                        documents, 
                        self.embeddings,
                        persist_directory=index_path
                    )
                    vector_store.persist()
                    logger.info(f"Chroma向量存储已创建并保存到: {index_path}")
                except Exception as e:
                    logger.error(f"Chroma创建失败: {str(e)}, 尝试备选方案")
                    # 尝试不使用persist_directory方式创建
                    vector_store = Chroma.from_documents(documents, self.embeddings)
                    logger.warning("创建了内存中的Chroma向量存储")
            else:
                logger.error(f"不支持的向量存储类型: {self.vector_store_type}")
                # 自动使用DocArray作为备选
                logger.info("尝试使用DocArrayInMemorySearch作为备选向量存储")
                vector_store = DocArrayInMemorySearch.from_documents(
                    documents, 
                    self.embeddings
                )
                logger.info("成功创建DocArrayInMemorySearch向量存储作为备选")
                
            return vector_store
            
        except Exception as e:
            logger.error(f"创建向量存储失败: {str(e)}")
            
            # 如果FAISS失败，尝试使用简单的向量存储实现
            logger.info("尝试创建备选向量存储...")
            try:
                # 使用DocArrayInMemorySearch作为备选
                backup_store = DocArrayInMemorySearch.from_documents(
                    documents, 
                    self.embeddings
                )
                logger.info("成功创建DocArray内存向量存储作为备选")
                return backup_store
            except Exception as inner_e:
                logger.error(f"创建备选向量存储也失败: {str(inner_e)}")
                return None
    
    @time_function
    def load_vector_store(
        self, 
        index_name: str = "default", 
        fallback_to_empty: bool = True
    ) -> Union[FAISS, Chroma, DocArrayInMemorySearch, None]:
        """
        加载向量存储
        
        Args:
            index_name: 索引名称
            fallback_to_empty: 如果为True，在索引不存在时返回空的内存向量存储
            
        Returns:
            向量存储对象
        """
        # 尝试确定索引路径
        # 首先尝试原始路径
        original_path = os.path.join(self.vector_store_dir, index_name)
        
        # 尝试可能的路径列表
        possible_paths = [
            original_path,  # 原始路径
            os.path.join(os.path.expanduser("~"), "rag_vector_store", index_name),  # 用户主目录
            os.path.join(tempfile.gettempdir(), "rag_vector_store", index_name),  # 临时目录
            os.path.join(os.getcwd(), "rag_vector_store", index_name),  # 当前目录
        ]
        
        # 检查每个可能的路径
        vector_store = None
        last_error = None
        
        for path in possible_paths:
            logger.info(f"尝试从路径加载向量存储: {path}")
            
            if not os.path.exists(path):
                logger.info(f"路径不存在，跳过: {path}")
                continue
                
            try:
                if self.vector_store_type == "faiss":
                    vector_store = FAISS.load_local(
                        path, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"成功从 {path} 加载FAISS向量存储")
                    break
                elif self.vector_store_type == "chroma":
                    vector_store = Chroma(
                        persist_directory=path,
                        embedding_function=self.embeddings
                    )
                    logger.info(f"成功从 {path} 加载Chroma向量存储")
                    break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"从 {path} 加载失败: {last_error}")
        
        # 如果所有路径都失败
        if vector_store is None:
            if fallback_to_empty:
                logger.warning(f"无法加载向量存储，创建空的内存向量存储: {last_error}")
                vector_store = DocArrayInMemorySearch.from_texts(
                    ["这是一个空的内存向量存储，请先添加文档。"], 
                    self.embeddings
                )
            else:
                logger.error(f"加载向量存储失败: {last_error}")
        
        return vector_store
    
    @time_function        
    def add_documents(
        self, 
        documents: List[Document], 
        vector_store: Union[FAISS, Chroma, DocArrayInMemorySearch],
        index_name: str = "default"
    ) -> Union[FAISS, Chroma, DocArrayInMemorySearch, None]:
        """
        向向量存储添加文档
        
        Args:
            documents: 文档列表
            vector_store: 向量存储对象
            index_name: 索引名称
            
        Returns:
            更新后的向量存储对象
        """
        if not documents:
            logger.warning("没有文档可添加")
            return vector_store
            
        if vector_store is None:
            logger.info("向量存储为空，创建新的向量存储")
            return self.create_vector_store(documents, index_name)
            
        logger.info(f"向向量存储添加 {len(documents)} 个文档")
        
        try:
            if isinstance(vector_store, FAISS):
                vector_store.add_documents(documents)
                index_path = os.path.join(self.vector_store_dir, index_name)
                vector_store.save_local(index_path)
                logger.info(f"文档已添加到FAISS向量存储并保存")
            elif isinstance(vector_store, Chroma):
                vector_store.add_documents(documents)
                vector_store.persist()
                logger.info(f"文档已添加到Chroma向量存储并保存")
            elif isinstance(vector_store, DocArrayInMemorySearch):
                # DocArrayInMemorySearch也支持add_documents方法
                vector_store.add_documents(documents)
                logger.info(f"文档已添加到DocArrayInMemorySearch向量存储")
            else:
                logger.error(f"不支持的向量存储类型: {type(vector_store)}")
                # 尝试创建新的向量存储
                logger.info("尝试创建新的DocArrayInMemorySearch向量存储")
                all_docs = documents
                if hasattr(vector_store, "get") and callable(getattr(vector_store, "get")):
                    try:
                        existing_docs = vector_store.get()["documents"]
                        all_docs = existing_docs + documents
                    except:
                        pass
                
                new_store = DocArrayInMemorySearch.from_documents(
                    all_docs, 
                    self.embeddings
                )
                logger.info(f"创建了新的DocArrayInMemorySearch向量存储")
                return new_store
                
            return vector_store
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {str(e)}")
            # 出错时尝试创建全新的向量存储
            try:
                logger.info("尝试创建全新的向量存储")
                return DocArrayInMemorySearch.from_documents(
                    documents, 
                    self.embeddings
                )
            except Exception as inner_e:
                logger.error(f"创建全新向量存储也失败: {str(inner_e)}")
                return vector_store

def get_vectorizer(
    embedding_model_name: str = EMBEDDING_MODEL,
    vector_store_dir: str = str(VECTOR_STORE_DIR),
    vector_store_type: str = "faiss"
) -> Vectorizer:
    """
    获取向量化器实例
    
    Args:
        embedding_model_name: 嵌入模型名称
        vector_store_dir: 向量存储目录
        vector_store_type: 向量存储类型
        
    Returns:
        Vectorizer实例
    """
    return Vectorizer(
        embedding_model_name=embedding_model_name,
        vector_store_dir=vector_store_dir,
        vector_store_type=vector_store_type
    ) 
"""
检索模块，用于根据用户查询检索相关文档片段
"""
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Set
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS, Chroma
from custom_retrievers import ContextualCompressionRetriever, LLMChainExtractor
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from utils.helpers import time_function
from config.config import TOP_K

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-Encoder重排序器，用于对检索结果进行精确重排序"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        初始化Cross-Encoder重排序器
        
        Args:
            model_name: Cross-Encoder模型名称
        """
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"Cross-Encoder重排序器初始化完成，模型: {model_name}")
        except Exception as e:
            logger.error(f"Cross-Encoder重排序器初始化失败: {str(e)}")
            self.model = None
            
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 保留的top_k个结果
            
        Returns:
            重排序后的文档列表
        """
        if not documents or not query or self.model is None:
            return documents[:top_k] if documents else []
            
        try:
            # 创建查询-文档对
            pairs = [(query, doc.page_content) for doc in documents]
            
            # 使用Cross-Encoder计算精确的相关性分数
            scores = self.model.predict(pairs)
            
            # 根据分数重新排序
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回得分最高的top_k个文档
            reranked_docs = [doc for doc, _ in doc_score_pairs[:top_k]]
            
            logger.info(f"Cross-Encoder重排序完成，从{len(documents)}个文档中选择了{len(reranked_docs)}个")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Cross-Encoder重排序失败: {str(e)}")
            return documents[:top_k]

class BM25Retriever:
    """BM25关键词检索器类"""
    
    def __init__(self, documents: List[Document], top_k: int = TOP_K):
        """
        初始化BM25检索器
        
        Args:
            documents: 文档列表
            top_k: 检索结果数量
        """
        self.documents = documents
        self.top_k = top_k
        self.tokenizer = self._simple_tokenizer
        
        # 提取文档文本内容
        self.texts = [doc.page_content for doc in documents]
        
        # 分词
        self.tokenized_texts = [self.tokenizer(text) for text in self.texts]
        
        # 创建BM25索引
        self.bm25 = BM25Okapi(self.tokenized_texts)
        
        logger.info(f"BM25检索器初始化完成，索引了 {len(documents)} 个文档")
        
    def _simple_tokenizer(self, text: str) -> List[str]:
        """
        简单的分词器，适用于中英文混合文本
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果
        """
        # 预处理：移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 中文分词：简单按字切分
        chinese_tokens = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 判断是否为中文字符
                chinese_tokens.append(char)
        
        # 英文分词：按空格分隔
        english_tokens = re.findall(r'[a-zA-Z0-9]+', text)
        
        # 合并分词结果
        tokens = chinese_tokens + english_tokens
        return tokens
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        根据查询检索相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        if not query:
            return []
            
        # 对查询进行分词
        tokenized_query = self.tokenizer(query)
        
        # 使用BM25计算相关性分数
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取排序后的索引
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        
        # 获取对应的文档
        retrieved_docs = [self.documents[i] for i in top_indices if scores[i] > 0]
        
        return retrieved_docs[:self.top_k]

class HybridRetriever:
    """混合检索器类，结合向量检索和关键词检索"""
    
    def __init__(
        self, 
        vector_retriever,
        bm25_retriever,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = TOP_K,
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        初始化混合检索器
        
        Args:
            vector_retriever: 向量检索器
            bm25_retriever: BM25检索器
            vector_weight: 向量检索结果权重
            bm25_weight: BM25检索结果权重
            top_k: 检索结果数量
            use_reranker: 是否使用Cross-Encoder重排序
            reranker_model: 重排序模型名称
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.use_reranker = use_reranker
        
        # 初始化重排序器
        if use_reranker:
            try:
                self.reranker = CrossEncoderReranker(model_name=reranker_model)
                logger.info("混合检索器启用了Cross-Encoder重排序")
            except Exception as e:
                logger.error(f"初始化Cross-Encoder重排序器失败: {str(e)}")
                self.use_reranker = False
                self.reranker = None
        else:
            self.reranker = None
        
        logger.info(f"混合检索器初始化完成，向量权重: {vector_weight}, BM25权重: {bm25_weight}")
        
    def _merge_documents(
        self, 
        vector_docs: List[Document], 
        bm25_docs: List[Document]
    ) -> List[Document]:
        """
        合并检索结果，去重并排序
        
        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            
        Returns:
            合并后的文档列表
        """
        # 使用集合追踪已经看到的内容以去重
        seen_contents: Set[str] = set()
        merged_docs: List[Document] = []
        
        # 从向量检索结果添加文档
        for doc in vector_docs:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                merged_docs.append(doc)
        
        # 从BM25检索结果添加文档
        for doc in bm25_docs:
            content = doc.page_content
            if content not in seen_contents:
                seen_contents.add(content)
                merged_docs.append(doc)
        
        # 限制返回数量
        return merged_docs[:max(self.top_k * 2, 10)]  # 为重排序保留更多候选
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        根据查询检索相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        if not query:
            return []
        
        # 获取向量检索结果
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        
        # 获取BM25检索结果
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        
        # 合并结果获得候选文档
        candidate_docs = self._merge_documents(vector_docs, bm25_docs)
        
        # 如果启用了重排序且有候选文档，则进行重排序
        if self.use_reranker and self.reranker and candidate_docs:
            final_docs = self.reranker.rerank(query, candidate_docs, self.top_k)
            return final_docs
        else:
            # 不使用重排序，直接返回合并结果的top_k个
            return candidate_docs[:self.top_k]

class Retriever:
    """检索器类，用于从向量库中检索相关文档"""
    
    def __init__(
        self, 
        vector_store: Union[FAISS, Chroma, 'SimpleMemoryVectorStore', Any],
        top_k: int = TOP_K,
        use_compression: bool = False,
        use_hybrid: bool = True,
        use_reranker: bool = True,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm = None
    ):
        """
        初始化检索器
        
        Args:
            vector_store: 向量存储，支持FAISS、Chroma或SimpleMemoryVectorStore
            top_k: 检索结果数量
            use_compression: 是否使用上下文压缩
            use_hybrid: 是否使用混合检索
            use_reranker: 是否使用重排序
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            reranker_model: 重排序模型名称
            llm: 语言模型实例（用于上下文压缩）
        """
        self.top_k = top_k
        self.use_compression = use_compression
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        
        # 支持SimpleMemoryVectorStore
        self.is_simple_memory_store = False
        if hasattr(vector_store, '__class__') and vector_store.__class__.__name__ == 'SimpleMemoryVectorStore':
            self.is_simple_memory_store = True
            self.simple_vector_store = vector_store
            
            # 获取所有文档，用于创建BM25检索器
            self.all_docs = vector_store.documents
            logger.info(f"检测到SimpleMemoryVectorStore，包含 {len(self.all_docs)} 个文档")
            
            # 创建基础检索器
            def vector_search_func(query, k=top_k):
                return vector_store.similarity_search(query, k)
                
            self.vector_retriever = self._create_func_retriever(vector_search_func, top_k)
        else:
            # 标准向量存储处理
            try:
                # 保存向量存储的引用
                self.vector_store = vector_store
                
                # 创建基础检索器
                self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
                
                # 获取所有文档，用于创建BM25检索器
                try:
                    self.all_docs = self._get_all_docs_from_vector_store(vector_store)
                except Exception as e:
                    logger.error(f"从向量存储获取所有文档失败: {str(e)}")
                    self.all_docs = []
            except Exception as e:
                logger.error(f"创建向量检索器失败: {str(e)}")
                raise
        
        # 获取所有文档用于BM25索引
        if use_hybrid:
            try:
                all_docs = vector_store.similarity_search("", k=1000)  # 获取足够多的文档建立BM25索引
                # 创建BM25检索器
                self.bm25_retriever = BM25Retriever(all_docs, top_k=top_k)
                
                # 创建混合检索器
                self.hybrid_retriever = HybridRetriever(
                    vector_retriever=self.vector_retriever,
                    bm25_retriever=self.bm25_retriever,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    top_k=top_k,
                    use_reranker=use_reranker,
                    reranker_model=reranker_model
                )
                logger.info("混合检索器初始化完成")
            except Exception as e:
                logger.error(f"初始化混合检索器失败: {str(e)}")
                self.use_hybrid = False
                logger.warning("回退到基础向量检索")
        
        # 直接使用Cross-Encoder重排序（如果不使用混合检索）
        if use_reranker and not use_hybrid:
            try:
                self.reranker = CrossEncoderReranker(model_name=reranker_model)
                logger.info("Cross-Encoder重排序器初始化完成")
            except Exception as e:
                logger.error(f"初始化Cross-Encoder重排序器失败: {str(e)}")
                self.reranker = None
                self.use_reranker = False
        
        # 设置压缩检索器(如果需要)
        if use_compression and llm is not None:
            compressor = LLMChainExtractor.from_llm(llm)
            if use_hybrid and hasattr(self, 'hybrid_retriever'):
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.hybrid_retriever
                )
            else:
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.vector_retriever
                )
            logger.info("初始化压缩检索器")
        else:
            # 使用混合或基础检索器
            if use_hybrid and hasattr(self, 'hybrid_retriever'):
                self.retriever = self.hybrid_retriever
                logger.info("使用混合检索器(含Cross-Encoder重排序)" if use_reranker else "使用混合检索器")
            else:
                self.retriever = self.vector_retriever
                logger.info("使用基础向量检索器")
            
        logger.info(f"检索器初始化完成，检索数量: {top_k}")
    
    @time_function
    def retrieve(self, query: str) -> List[Document]:
        """
        根据查询检索相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        if not query:
            logger.warning("查询为空")
            return []
            
        logger.info(f"执行查询: {query[:50]}...")
        
        try:
            # 使用配置的检索器获取文档
            docs = self.retriever.get_relevant_documents(query)
            
            # 如果没有使用混合检索但启用了重排序，在这里应用重排序
            if not self.use_hybrid and self.use_reranker and hasattr(self, 'reranker') and self.reranker:
                # 获取更多候选进行重排序
                candidate_docs = self.vector_retriever.get_relevant_documents(query)
                docs = self.reranker.rerank(query, candidate_docs, self.top_k)
            
            logger.info(f"检索到 {len(docs)} 个相关文档")
            return docs
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return []
    
    @time_function
    def retrieve_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        """
        根据查询检索相关文档及其相似度分数
        
        Args:
            query: 查询文本
            
        Returns:
            (文档,相似度分数)元组列表
        """
        if not query:
            logger.warning("查询为空")
            return []
            
        if not hasattr(self.vector_store, "similarity_search_with_score"):
            logger.warning("向量存储不支持带分数的相似度搜索")
            return [(doc, 0.0) for doc in self.retrieve(query)]
            
        logger.info(f"执行带分数查询: {query[:50]}...")
        
        try:
            # 注意：混合检索目前不支持直接返回分数，此处仍使用向量存储的相似度搜索
            results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
            logger.info(f"检索到 {len(results)} 个相关文档及分数")
            return results
        except Exception as e:
            logger.error(f"带分数检索失败: {str(e)}")
            return []
            
    def update_vector_store(self, vector_store: Union[FAISS, Chroma]) -> None:
        """
        更新向量存储
        
        Args:
            vector_store: 新的向量存储对象
        """
        if vector_store is None:
            logger.error("新的向量存储为空，无法更新")
            return
            
        self.vector_store = vector_store
        self.vector_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # 如果使用混合检索，需要更新BM25检索器
        if self.use_hybrid:
            try:
                all_docs = vector_store.similarity_search("", k=1000)
                self.bm25_retriever = BM25Retriever(all_docs, top_k=self.top_k)
                self.hybrid_retriever = HybridRetriever(
                    vector_retriever=self.vector_retriever,
                    bm25_retriever=self.bm25_retriever,
                    vector_weight=self.vector_weight,
                    bm25_weight=self.bm25_weight,
                    top_k=self.top_k,
                    use_reranker=self.use_reranker,
                    reranker_model=self.reranker_model
                )
            except Exception as e:
                logger.error(f"更新混合检索器失败: {str(e)}")
        
        if self.use_compression and self.llm is not None:
            compressor = LLMChainExtractor.from_llm(self.llm)
            if self.use_hybrid and hasattr(self, 'hybrid_retriever'):
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.hybrid_retriever
                )
            else:
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.vector_retriever
                )
        else:
            if self.use_hybrid and hasattr(self, 'hybrid_retriever'):
                self.retriever = self.hybrid_retriever
            else:
                self.retriever = self.vector_retriever
            
        logger.info("检索器向量存储已更新")
        
    def _create_func_retriever(self, search_func: Callable, top_k: int) -> Any:
        """
        创建基于函数的检索器
        
        Args:
            search_func: 搜索函数
            top_k: 检索结果数量
            
        Returns:
            函数检索器对象
        """
        class FuncRetriever:
            def __init__(self, func, k):
                self.func = func
                self.k = k
                
            def get_relevant_documents(self, query):
                return self.func(query, self.k)
                
        return FuncRetriever(search_func, top_k)
        
    def _get_all_docs_from_vector_store(self, vector_store) -> List[Document]:
        """
        从向量存储中获取所有文档
        
        Args:
            vector_store: 向量存储
            
        Returns:
            文档列表
        """
        # 针对不同类型的向量存储采用不同的获取方法
        if isinstance(vector_store, FAISS):
            # FAISS存储了文档的拷贝
            all_docs = []
            # 使用一个非常宽泛的查询获取尽可能多的文档
            try:
                # 尝试通过向量空间范围查询获取
                for i in range(10):  # 多次查询以获取更多文档
                    query = f"document {i}"
                    docs = vector_store.similarity_search(query, k=1000)
                    all_docs.extend(docs)
                # 去重
                seen_contents = set()
                unique_docs = []
                for doc in all_docs:
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        unique_docs.append(doc)
                return unique_docs
            except Exception as e:
                logger.error(f"无法从FAISS获取所有文档: {str(e)}")
                return []
                
        elif isinstance(vector_store, Chroma):
            # Chroma可以通过get方法获取所有文档
            try:
                return vector_store.get()["documents"]
            except Exception as e:
                logger.error(f"无法从Chroma获取所有文档: {str(e)}")
                return []
                
        else:
            # 对于其他类型，尝试通用方法
            try:
                if hasattr(vector_store, "get"):
                    return vector_store.get()["documents"]
                elif hasattr(vector_store, "similarity_search"):
                    # 尝试一个通用查询
                    return vector_store.similarity_search("", k=1000)
                else:
                    logger.error(f"不支持的向量存储类型: {type(vector_store)}")
                    return []
            except Exception as e:
                logger.error(f"获取文档失败: {str(e)}")
                return []

def get_retriever(
    vector_store: Union[FAISS, Chroma, 'SimpleMemoryVectorStore', Any],
    top_k: int = TOP_K,
    use_compression: bool = False,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm = None
) -> Retriever:
    """
    获取检索器实例
    
    Args:
        vector_store: 向量存储，支持FAISS、Chroma或SimpleMemoryVectorStore
        top_k: 检索结果数量
        use_compression: 是否使用上下文压缩
        use_hybrid: 是否使用混合检索
        use_reranker: 是否使用Cross-Encoder重排序
        vector_weight: 向量检索权重
        bm25_weight: BM25检索权重
        reranker_model: 重排序模型名称
        llm: 语言模型实例（用于上下文压缩）
        
    Returns:
        Retriever实例
    """
    return Retriever(
        vector_store=vector_store,
        top_k=top_k,
        use_compression=use_compression,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        reranker_model=reranker_model,
        llm=llm
    ) 
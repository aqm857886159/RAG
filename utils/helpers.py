"""
辅助函数模块，包含各种通用工具函数
"""
import os
import time
import logging
import functools
from typing import List, Callable, Any, Dict, Optional
from pathlib import Path

from langchain.docstore.document import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def time_function(func: Callable) -> Callable:
    """
    计时装饰器，用于记录函数执行时间
    
    Args:
        func: 要计时的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        logger.info(f"函数 {func.__name__} 执行时间: {elapsed_time:.4f} 秒")
        
        return result
    return wrapper

def ensure_directory(directory_path: str) -> str:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory_path: 目录路径
        
    Returns:
        创建的目录路径
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {path}")
    return str(path)

def format_source_documents(docs: List[Document]) -> str:
    """
    格式化源文档信息，用于展示
    
    Args:
        docs: 文档列表
        
    Returns:
        格式化后的文档信息
    """
    if not docs:
        return "无来源文档"
    
    formatted_docs = []
    for i, doc in enumerate(docs):
        # 提取元数据
        metadata = doc.metadata or {}
        source = metadata.get("source", "未知来源")
        page = metadata.get("page", "")
        page_info = f" (页码: {page})" if page else ""
        
        # 构建文档信息
        doc_info = f"来源 {i+1}: {source}{page_info}\n\n{doc.page_content}\n"
        formatted_docs.append(doc_info)
    
    return "\n--------------------\n".join(formatted_docs)

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    截断文本，超过最大长度则添加省略号
    
    Args:
        text: 原始文本
        max_length: 最大长度
        
    Returns:
        截断后的文本
    """
    if not text:
        return ""
        
    if len(text) <= max_length:
        return text
        
    return text[:max_length] + "..."

def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件扩展名(小写)
    """
    return os.path.splitext(file_path)[1].lower()

def safe_get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    """
    安全获取对象属性，如果不存在则返回默认值
    
    Args:
        obj: 目标对象
        attr_name: 属性名
        default: 默认值
        
    Returns:
        属性值或默认值
    """
    try:
        return getattr(obj, attr_name, default)
    except:
        return default

def list_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    列出目录中指定扩展名的所有文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，如['.pdf', '.txt']
        
    Returns:
        文件路径列表
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if extensions is None or get_file_extension(file_path) in extensions:
                files.append(file_path)
    return files

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分割成指定大小的块
    
    Args:
        lst: 要分割的列表
        chunk_size: 每个块的大小
        
    Returns:
        分割后的列表的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)] 
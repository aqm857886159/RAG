"""
辅助函数模块，提供各种实用工具函数
"""
import os
import logging
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_file_extension(file_path: str) -> str:
    """
    获取文件扩展名
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件扩展名(小写)
    """
    return os.path.splitext(file_path)[1].lower()

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

def time_function(func):
    """
    函数执行时间装饰器
    
    Args:
        func: 要计时的函数
        
    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"函数 {func.__name__} 执行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper

def ensure_directory(directory: str) -> None:
    """
    确保目录存在，不存在则创建
    支持Windows路径和非ASCII字符
    
    Args:
        directory: 目录路径
    """
    try:
        # 规范化路径，处理斜杠和反斜杠
        norm_path = os.path.normpath(directory)
        # 使用pathlib，处理递归创建和权限问题
        path_obj = Path(norm_path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # 验证目录是否可写
        if not os.access(norm_path, os.W_OK):
            logger.warning(f"目录 {norm_path} 存在但不可写，尝试修复权限...")
            # 尝试在目录中创建临时文件测试写入权限
            try:
                test_file = os.path.join(norm_path, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                logger.info(f"目录 {norm_path} 可写")
            except Exception as e:
                logger.error(f"目录 {norm_path} 不可写: {str(e)}")
                raise PermissionError(f"目录 {norm_path} 不可写，请检查权限")
        
        logger.info(f"确保目录存在: {norm_path}")
    except Exception as e:
        logger.error(f"创建目录失败: {directory}, 错误: {str(e)}")
        raise

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

def format_source_documents(source_documents: List[Any]) -> str:
    """
    格式化源文档为可读文本
    
    Args:
        source_documents: 源文档列表
        
    Returns:
        格式化后的文本
    """
    if not source_documents:
        return "没有找到相关文档"
        
    formatted_docs = []
    for i, doc in enumerate(source_documents):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content
        formatted_docs.append(f"文档[{i+1}] 来源: {source}\n内容: {content}\n")
        
    return "\n".join(formatted_docs) 
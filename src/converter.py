"""
文档转换模块，用于将各种格式的文档转换为纯文本
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_file(file_path: str, **kwargs) -> str:
    """
    将文档转换为纯文本
    
    Args:
        file_path: 文件路径
        kwargs: 额外参数
        
    Returns:
        转换后的文本内容
    """
    file_path = Path(file_path)
    file_extension = file_path.suffix.lower()
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        if file_extension == '.pdf':
            return convert_pdf(str(file_path), **kwargs)
        elif file_extension == '.docx':
            return convert_docx(str(file_path), **kwargs)
        elif file_extension == '.txt':
            return convert_txt(str(file_path), **kwargs)
        else:
            logger.error(f"不支持的文件格式: {file_extension}")
            raise ValueError(f"不支持的文件格式: {file_extension}")
    except Exception as e:
        logger.error(f"转换文件失败: {str(e)}")
        raise

def convert_pdf(file_path: str, **kwargs) -> str:
    """
    转换PDF文件为纯文本
    
    Args:
        file_path: PDF文件路径
        kwargs: 额外参数
        
    Returns:
        PDF文件的文本内容
    """
    try:
        import PyPDF2
        
        logger.info(f"开始转换PDF文件: {file_path}")
        
        with open(file_path, 'rb') as pdf_file:
            # 创建PDF读取器
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # 获取页数
            num_pages = len(pdf_reader.pages)
            logger.info(f"PDF文件共有 {num_pages} 页")
            
            # 遍历所有页面并提取文本
            for i in range(num_pages):
                page = pdf_reader.pages[i]
                page_text = page.extract_text()
                
                if page_text:
                    text += page_text + "\n\n"
                
            logger.info(f"PDF转换完成，文本长度: {len(text)}")
            return text
    except ImportError:
        logger.error("缺少PyPDF2库，请安装: pip install PyPDF2")
        raise
    except Exception as e:
        logger.error(f"转换PDF文件时出错: {str(e)}")
        raise

def convert_docx(file_path: str, **kwargs) -> str:
    """
    转换DOCX文件为纯文本
    
    Args:
        file_path: DOCX文件路径
        kwargs: 额外参数
        
    Returns:
        DOCX文件的文本内容
    """
    try:
        import docx
        
        logger.info(f"开始转换DOCX文件: {file_path}")
        
        # 打开DOCX文件
        doc = docx.Document(file_path)
        
        # 提取段落文本
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
        
        logger.info(f"DOCX转换完成，文本长度: {len(text)}")
        return text
    except ImportError:
        logger.error("缺少python-docx库，请安装: pip install python-docx")
        raise
    except Exception as e:
        logger.error(f"转换DOCX文件时出错: {str(e)}")
        raise

def convert_txt(file_path: str, encoding: str = 'utf-8', **kwargs) -> str:
    """
    读取TXT文件内容
    
    Args:
        file_path: TXT文件路径
        encoding: 文件编码
        kwargs: 额外参数
        
    Returns:
        TXT文件的文本内容
    """
    logger.info(f"开始读取TXT文件: {file_path}")
    
    try:
        # 尝试使用指定编码打开文件
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
            
        logger.info(f"TXT文件读取完成，文本长度: {len(text)}")
        return text
    except UnicodeDecodeError:
        logger.warning(f"使用 {encoding} 编码读取失败，尝试其他编码")
        
        # 尝试其他常见编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'ascii']
        for enc in encodings:
            if enc != encoding:  # 跳过已尝试的编码
                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        text = f.read()
                    logger.info(f"成功使用 {enc} 编码读取TXT文件")
                    return text
                except UnicodeDecodeError:
                    continue
        
        # 如果所有编码都失败，尝试二进制读取
        logger.warning("所有编码均失败，尝试二进制读取")
        with open(file_path, 'rb') as f:
            text = f.read().decode('utf-8', errors='replace')
        
        logger.info(f"二进制读取完成，文本长度: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"读取TXT文件时出错: {str(e)}")
        raise 
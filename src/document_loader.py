"""
文档加载模块，支持多种格式文档的加载和处理
"""
import os
import argparse
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import PyPDF2
import docx2txt
import pdfplumber
from langchain_community.docstore.document import Document

from custom_loaders import CustomTextLoader, CustomPDFLoader, CustomDocxLoader, CustomCSVLoader, CustomExcelLoader
from utils.helpers import get_file_extension, list_files, time_function
from config.config import DATA_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentLoader:
    """文档加载器类，支持多种格式文档的加载和处理"""
    
    def __init__(self, input_dir: Optional[str] = None):
        """
        初始化文档加载器
        
        Args:
            input_dir: 输入文档目录，默认为配置文件中的DATA_DIR
        """
        self.input_dir = input_dir or str(DATA_DIR)
        logger.info(f"初始化文档加载器，输入目录: {self.input_dir}")
        
    @time_function
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        加载单个文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            Document对象列表
        """
        ext = get_file_extension(file_path)
        try:
            if ext == '.pdf':
                loader = CustomPDFLoader(file_path)
                documents = loader.load()
            elif ext == '.docx' or ext == '.doc':
                loader = CustomDocxLoader(file_path)
                documents = loader.load()
            elif ext == '.txt':
                loader = CustomTextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif ext == '.csv':
                loader = CustomCSVLoader(file_path)
                documents = loader.load()
            elif ext in ['.xlsx', '.xls']:
                # 使用自定义Excel加载器
                loader = CustomExcelLoader(file_path)
                documents = loader.load()
            else:
                logger.warning(f"不支持的文件类型: {ext}, 文件: {file_path}")
                return []
                
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个片段")
            # 添加文件路径到元数据
            for doc in documents:
                doc.metadata["source"] = file_path
                
            return documents
        except Exception as e:
            logger.error(f"加载文档失败: {file_path}, 错误: {str(e)}")
            return []
    
    @time_function
    def load_documents(self, file_paths: Optional[List[str]] = None) -> List[Document]:
        """
        加载多个文档
        
        Args:
            file_paths: 文档路径列表，如果为None则加载input_dir中的所有支持格式文档
            
        Returns:
            Document对象列表
        """
        if file_paths is None:
            file_paths = list_files(
                self.input_dir, 
                extensions=['.pdf', '.docx', '.doc', '.txt', '.csv', '.xlsx', '.xls']
            )
        
        if not file_paths:
            logger.warning(f"没有找到可加载的文档，路径: {self.input_dir}")
            return []
            
        logger.info(f"准备加载 {len(file_paths)} 个文档")
        all_documents = []
        
        for file_path in file_paths:
            documents = self.load_single_document(file_path)
            all_documents.extend(documents)
            
        logger.info(f"文档加载完成，共 {len(all_documents)} 个片段")
        return all_documents

    @time_function
    def load_excel_file(self, file_path: str) -> List[Document]:
        """
        加载Excel文件
        
        Args:
            file_path: Excel文件路径
            
        Returns:
            Document对象列表
        """
        try:
            import pandas as pd
            
            logger.info(f"开始加载Excel文件: {file_path}")
            # 尝试使用calamine引擎（如果可用）
            try:
                all_sheets_data = pd.read_excel(file_path, sheet_name=None, engine='calamine')
            except Exception:
                # 如果calamine不可用，回退到默认引擎
                all_sheets_data = pd.read_excel(file_path, sheet_name=None)
                
            documents = []
            
            for sheet_name, df in all_sheets_data.items():
                logger.info(f"处理工作表: {sheet_name}, 形状: {df.shape}")
                
                # 转换DataFrame为文本
                text_content = f"工作表: {sheet_name}\n\n"
                text_content += df.to_string(index=False)
                
                # 创建Document对象
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "source": file_path,
                        "sheet_name": sheet_name,
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "file_type": "excel"
                    }
                )
                
                # 将表格数据存储为JSON可序列化的格式
                table_data = df.replace({pd.NaT: None}).to_dict(orient='records')
                doc.metadata["table_data"] = table_data
                
                documents.append(doc)
                
            logger.info(f"Excel文件加载完成: {file_path}, 共 {len(documents)} 个工作表")
            return documents
            
        except Exception as e:
            logger.error(f"加载Excel文件失败: {file_path}, 错误: {str(e)}")
            return []
            
    @time_function
    def load_with_ocr(self, file_path: str) -> List[Document]:
        """
        使用OCR加载文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            Document对象列表
        """
        try:
            # 检查文件类型
            ext = get_file_extension(file_path)
            
            if ext == '.pdf':
                logger.info(f"使用OCR处理PDF文件: {file_path}")
                
                try:
                    # 尝试导入必要的库
                    from paddleocr import PaddleOCR
                    import fitz  # PyMuPDF
                    import numpy as np
                    from PIL import Image
                    import io
                    
                    # 初始化PaddleOCR
                    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
                    
                    # 打开PDF
                    doc = fitz.open(file_path)
                    documents = []
                    
                    # 处理每一页
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        
                        # 渲染页面为图像
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        
                        # 将pixmap转换为PIL图像
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # 创建临时内存文件
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format="PNG")
                        img_bytes.seek(0)
                        
                        # OCR处理
                        result = ocr.ocr(img_bytes, cls=True)
                        
                        # 提取文本
                        text = ""
                        if result is not None and len(result) > 0:
                            for line in result:
                                for word_info in line:
                                    if len(word_info) >= 2 and isinstance(word_info[1], tuple) and len(word_info[1]) >= 1:
                                        text += word_info[1][0] + " "
                                text += "\n"
                        
                        # 创建Document
                        if text.strip():  # 只有在有文本的情况下才添加文档
                            documents.append(Document(
                                page_content=text,
                                metadata={
                                    "source": file_path, 
                                    "page": page_num + 1,
                                    "file_type": "pdf",
                                    "ocr_processed": True
                                }
                            ))
                            
                    logger.info(f"OCR处理完成: {file_path}, 共 {len(documents)} 页")
                    return documents
                    
                except ImportError as e:
                    logger.error(f"OCR处理失败，缺少必要的库: {str(e)}")
                    logger.info("回退到标准PDF加载器")
                    return self.load_single_document(file_path)
            else:
                logger.warning(f"不支持OCR处理的文件类型: {ext}")
                return self.load_single_document(file_path)
                
        except Exception as e:
            logger.error(f"OCR处理失败: {file_path}, 错误: {str(e)}")
            # 回退到常规加载器
            logger.info("回退到标准文档加载器")
            return self.load_single_document(file_path)

def main():
    """主函数，处理命令行参数并执行文档加载"""
    parser = argparse.ArgumentParser(description="文档加载工具")
    parser.add_argument("--input_dir", type=str, default=str(DATA_DIR),
                      help="输入文档目录路径")
    parser.add_argument("--file", type=str, help="单个文件路径")
    parser.add_argument("--use_ocr", action="store_true", help="使用OCR处理文档")
    
    args = parser.parse_args()
    
    loader = DocumentLoader(args.input_dir)
    
    if args.file:
        if args.use_ocr and args.file.lower().endswith('.pdf'):
            documents = loader.load_with_ocr(args.file)
        else:
            documents = loader.load_single_document(args.file)
    else:
        documents = loader.load_documents()
        
    print(f"成功加载 {len(documents)} 个文档片段")
    
    if documents and len(documents) > 0:
        print("\n示例文档内容:")
        doc = documents[0]
        print(f"内容: {doc.page_content[:200]}...")
        print(f"元数据: {doc.metadata}")
        
if __name__ == "__main__":
    main() 
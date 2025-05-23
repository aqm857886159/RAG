"""
自定义文档加载器模块，避免使用langchain_community导入
"""
import os
import csv
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_community.docstore.document import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomTextLoader:
    """自定义文本文件加载器"""
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding
    
    def load(self) -> List[Document]:
        """加载文本文件内容"""
        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                text = f.read()
            
            metadata = {"source": self.file_path}
            return [Document(page_content=text, metadata=metadata)]
        except Exception as e:
            logger.error(f"加载文本文件失败: {self.file_path}, 错误: {str(e)}")
            return []

class CustomPDFLoader:
    """自定义PDF文件加载器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """加载PDF文件内容"""
        try:
            import PyPDF2
            
            documents = []
            with open(self.file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                for i, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():  # 只有在有文本的情况下才添加
                        metadata = {
                            "source": self.file_path,
                            "page": i + 1
                        }
                        documents.append(Document(page_content=text, metadata=metadata))
            
            return documents
        except Exception as e:
            logger.error(f"加载PDF文件失败: {self.file_path}, 错误: {str(e)}")
            return []

class CustomDocxLoader:
    """自定义Word文档加载器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """加载Word文档内容"""
        try:
            import docx2txt
            
            text = docx2txt.process(self.file_path)
            metadata = {"source": self.file_path}
            return [Document(page_content=text, metadata=metadata)]
        except Exception as e:
            logger.error(f"加载Word文档失败: {self.file_path}, 错误: {str(e)}")
            return []

class CustomCSVLoader:
    """自定义CSV文件加载器"""
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        self.file_path = file_path
        self.encoding = encoding
    
    def load(self) -> List[Document]:
        """加载CSV文件内容"""
        try:
            documents = []
            
            with open(self.file_path, newline="", encoding=self.encoding) as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader)
                
                for i, row in enumerate(csv_reader):
                    # 创建CSV行文本
                    content = "\n".join([f"{headers[j]}: {value}" for j, value in enumerate(row) if j < len(headers)])
                    
                    metadata = {
                        "source": self.file_path,
                        "row": i + 1,
                        "header": headers
                    }
                    
                    documents.append(Document(page_content=content, metadata=metadata))
            
            return documents
        except Exception as e:
            logger.error(f"加载CSV文件失败: {self.file_path}, 错误: {str(e)}")
            return []

class CustomExcelLoader:
    """自定义Excel文件加载器"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> List[Document]:
        """加载Excel文件内容"""
        try:
            import pandas as pd
            
            # 读取所有工作表
            all_sheets_data = pd.read_excel(self.file_path, sheet_name=None)
            documents = []
            
            for sheet_name, df in all_sheets_data.items():
                # 转换DataFrame为文本
                text_content = f"工作表: {sheet_name}\n\n"
                text_content += df.to_string(index=False)
                
                # 创建Document对象
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "source": self.file_path,
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
            
            return documents
        except Exception as e:
            logger.error(f"加载Excel文件失败: {self.file_path}, 错误: {str(e)}")
            return [] 
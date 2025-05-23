#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将指定文档添加到向量库的脚本
"""
import os
import sys
import logging
import shutil
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("开始执行文档添加脚本...")
print(f"当前工作目录: {os.getcwd()}")

from src.document_loader import DocumentLoader
from src.text_splitter import get_text_splitter
from src.vectorizer import get_vectorizer
from utils.helpers import ensure_directory
from config.config import DATA_DIR, VECTOR_STORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

print(f"配置信息加载完成:")
print(f"- 数据目录: {DATA_DIR}")
print(f"- 向量库目录: {VECTOR_STORE_DIR}")
print(f"- 块大小: {CHUNK_SIZE}")
print(f"- 重叠大小: {CHUNK_OVERLAP}")
print(f"- 嵌入模型: {EMBEDDING_MODEL}")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 文档源路径
        doc_path = os.path.join(os.getcwd(), "Untitled.docx")
        print(f"文档路径: {doc_path}")
        print(f"文档是否存在: {os.path.exists(doc_path)}")
        
        # 先清理可能有问题的向量库目录
        vector_store_path = str(VECTOR_STORE_DIR)
        default_index_path = os.path.join(vector_store_path, "default")
        
        if os.path.exists(default_index_path):
            print(f"删除现有向量库目录: {default_index_path}")
            shutil.rmtree(default_index_path, ignore_errors=True)
        
        # 确保目录存在
        ensure_directory(DATA_DIR)
        ensure_directory(vector_store_path)
        ensure_directory(default_index_path)
        print(f"目录创建完成: {DATA_DIR}, {vector_store_path}, {default_index_path}")
        
        # 加载文档
        print(f"正在加载文档...")
        loader = DocumentLoader()
        documents = loader.load_single_document(doc_path)
        print(f"文档加载完成，共 {len(documents)} 页")
        
        # 文本分块
        print(f"正在进行文本分块...")
        text_splitter = get_text_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
        print(f"文本分块完成，共 {len(chunks)} 个块")
        
        # 向量化并存储
        print(f"正在进行向量化...")
        vectorizer = get_vectorizer(
            embedding_model_name=EMBEDDING_MODEL,
            vector_store_dir=vector_store_path,
            vector_store_type="chroma"  # 尝试使用Chroma代替FAISS
        )
        
        # 创建新的向量库
        print(f"创建新的向量库...")
        vector_store = vectorizer.create_vector_store(chunks, index_name="default")
            
        if vector_store:
            print(f"向量化完成，文档已添加到向量库")
            print(f"处理成功完成！")
        else:
            print(f"向量化失败，请检查日志")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 
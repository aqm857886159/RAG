#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基本功能测试脚本 - 不依赖复杂的LLM模块
"""
import os
import sys
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).parent
    if current_dir.name == 'tests':
        return current_dir.parent
    return current_dir

def test_basic_imports():
    """测试基本模块导入"""
    logger.info("测试基本模块导入...")
    
    try:
        import streamlit as st
        logger.info("✓ Streamlit导入成功")
    except ImportError as e:
        logger.error(f"✗ Streamlit导入失败: {e}")
    
    try:
        from langchain.docstore.document import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        logger.info("✓ LangChain核心模块导入成功")
    except ImportError as e:
        logger.error(f"✗ LangChain导入失败: {e}")
    
    try:
        import docx2txt
        logger.info("✓ docx2txt导入成功")
    except ImportError as e:
        logger.error(f"✗ docx2txt导入失败: {e}")
    
    try:
        import pandas as pd
        logger.info("✓ pandas导入成功")
    except ImportError as e:
        logger.error(f"✗ pandas导入失败: {e}")

def test_document_loading():
    """测试文档加载功能"""
    logger.info("测试文档加载功能...")
    
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    # 查找测试文件
    test_files = []
    if data_dir.exists():
        for ext in ['.docx', '.pdf', '.txt', '.csv']:
            files = list(data_dir.glob(f"*{ext}"))
            test_files.extend(files)
    
    if not test_files:
        logger.error("未找到测试文件")
        return False
    
    logger.info(f"找到 {len(test_files)} 个测试文件")
    
    # 测试Word文档加载
    for file_path in test_files:
        if file_path.suffix == '.docx':
            try:
                import docx2txt
                content = docx2txt.process(str(file_path))
                logger.info(f"✓ Word文档加载成功: {file_path.name} ({len(content)} 字符)")
                return True
            except Exception as e:
                logger.error(f"✗ Word文档加载失败: {e}")
        
        elif file_path.suffix == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"✓ 文本文件加载成功: {file_path.name} ({len(content)} 字符)")
                return True
            except Exception as e:
                logger.error(f"✗ 文本文件加载失败: {e}")
    
    return False

def test_text_splitting():
    """测试文本分割功能"""
    logger.info("测试文本分割功能...")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # 创建测试文本
        test_text = "这是一个测试文档。" * 100
        
        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
        )
        
        # 分割文本
        chunks = text_splitter.split_text(test_text)
        
        logger.info(f"✓ 文本分割成功: {len(test_text)} 字符分割为 {len(chunks)} 个块")
        return True
        
    except Exception as e:
        logger.error(f"✗ 文本分割失败: {e}")
        return False

def test_data_structure_creation():
    """测试数据结构创建"""
    logger.info("测试数据结构创建...")
    
    try:
        # 创建模拟的结构化数据
        structured_data = {
            "chunks": [
                {
                    "content": "这是第一个文本块的内容",
                    "metadata": {
                        "source": "test_document.docx",
                        "chunk_id": 1
                    }
                },
                {
                    "content": "这是第二个文本块的内容",
                    "metadata": {
                        "source": "test_document.docx",
                        "chunk_id": 2
                    }
                }
            ],
            "metadata": {
                "total_chunks": 2,
                "source_file": "test_document.docx",
                "processing_time": "2024-01-01 12:00:00"
            }
        }
        
        # 保存到输出目录
        project_root = get_project_root()
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "test_structure.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 数据结构创建并保存成功: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ 数据结构创建失败: {e}")
        return False

def test_streamlit_components():
    """测试Streamlit组件（不启动服务器）"""
    logger.info("测试Streamlit组件...")
    
    try:
        import streamlit as st
        
        # 测试基本组件是否可用
        components = [
            'selectbox', 'file_uploader', 'text_input', 
            'number_input', 'checkbox', 'button'
        ]
        
        for component in components:
            if hasattr(st, component):
                logger.info(f"✓ Streamlit组件可用: {component}")
            else:
                logger.warning(f"? Streamlit组件不存在: {component}")
        
        logger.info("✓ Streamlit组件测试完成")
        return True
        
    except Exception as e:
        logger.error(f"✗ Streamlit组件测试失败: {e}")
        return False

def test_project_structure():
    """测试项目结构"""
    logger.info("测试项目结构...")
    
    project_root = get_project_root()
    logger.info(f"项目根目录: {project_root}")
    
    required_dirs = ['src', 'data', 'output', 'tests', 'config']
    required_files = ['requirements.txt', 'README.md', 'run_app.py']
    
    # 检查目录
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            logger.info(f"✓ 目录存在: {dir_name}")
        else:
            logger.warning(f"? 目录不存在: {dir_name}")
    
    # 检查文件
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            logger.info(f"✓ 文件存在: {file_name}")
        else:
            logger.warning(f"? 文件不存在: {file_name}")
    
    return True

def main():
    """主测试函数"""
    logger.info("开始基本功能测试...")
    
    test_results = {
        "项目结构": test_project_structure(),
        "基本导入": test_basic_imports(),
        "文档加载": test_document_loading(),
        "文本分割": test_text_splitting(),
        "数据结构": test_data_structure_creation(),
        "Streamlit组件": test_streamlit_components(),
    }
    
    logger.info("\n" + "="*50)
    logger.info("测试结果总结:")
    logger.info("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 所有基本功能测试通过！系统基础功能正常。")
    elif passed >= total * 0.8:
        logger.info("⚠️  大部分测试通过，系统基本可用，有少量问题需要解决。")
    else:
        logger.error("❌ 多项测试失败，需要检查环境配置和依赖安装。")

if __name__ == "__main__":
    main() 
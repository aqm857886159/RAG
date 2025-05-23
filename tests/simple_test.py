"""
简化的文档转换测试脚本
"""
import os
import logging
import json
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).parent
    # 从tests目录向上找到项目根目录
    if current_dir.name == 'tests':
        return current_dir.parent
    else:
        return current_dir

# 主函数
def main():
    """主函数"""
    # 获取项目根目录
    project_root = get_project_root()
    
    # 导入需要的模块
    try:
        from langchain.docstore.document import Document
        import streamlit as st
        import docx2txt
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        import pandas as pd
    except ImportError as e:
        logger.error(f"缺少必要的库: {e}")
        logger.info("请安装所需的依赖: pip install -r requirements.txt")
        return

    # 设置测试文件 - 使用项目根目录的相对路径
    file_path = project_root / "data" / "Untitled (1).docx"
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        logger.info(f"请确保文件位于: {file_path}")
        # 尝试查找其他可能的文件
        data_dir = project_root / "data"
        if data_dir.exists():
            logger.info("data目录中的文件:")
            for f in data_dir.iterdir():
                if f.suffix in ['.docx', '.pdf', '.txt']:
                    logger.info(f"  - {f.name}")
        return
    
    # 输出目录
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 加载文档
    logger.info(f"加载文档: {file_path}")
    content = docx2txt.process(str(file_path))
    logger.info(f"文档内容长度: {len(content)} 字符")
    
    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_text(content)
    logger.info(f"文本分割为 {len(chunks)} 个块")
    
    # 创建输出结构
    result = {
        "chunks": [{"content": chunk, "metadata": {"source": str(file_path)}} for chunk in chunks],
        "metadata": {
            "source": str(file_path),
            "chunk_count": len(chunks),
            "total_length": len(content)
        }
    }
    
    # 保存结果
    output_path = output_dir / "simple_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {output_path}")
    
    # 显示部分结果
    logger.info("结果示例:")
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"块 {i+1}: {chunk[:100]}...")
    
    if len(chunks) > 3:
        logger.info(f"还有 {len(chunks) - 3} 个块未显示")

if __name__ == "__main__":
    main() 
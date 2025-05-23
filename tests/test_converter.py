#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试文档转换功能
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

# 获取项目根目录并添加到Python路径
def get_project_root():
    """获取项目根目录"""
    current_dir = Path(__file__).parent
    # 从tests目录向上找到项目根目录
    if current_dir.name == 'tests':
        return current_dir.parent
    else:
        return current_dir

project_root = get_project_root()
sys.path.insert(0, str(project_root))

# 确保src目录在导入路径中
try:
    from src.data_converter import DocumentConverter
    from src.document_loader import DocumentLoader
    from config.config import DATA_DIR
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"项目根目录: {project_root}")
    print(f"确保以下目录存在并包含相应模块:")
    print(f"  - {project_root}/src/data_converter.py")
    print(f"  - {project_root}/src/document_loader.py")
    print(f"  - {project_root}/config/config.py")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试文档转换功能")
    parser.add_argument("--file", type=str, help="要转换的文档路径（可选，默认使用示例文档）")
    parser.add_argument("--chunk_size", type=int, default=500, help="文本块大小")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="块重叠大小")
    parser.add_argument("--use_ocr", action="store_true", help="使用OCR处理PDF")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "deepseek"], help="LLM提供商")
    parser.add_argument("--output_formats", type=str, nargs="+", default=["json"], help="输出格式")
    parser.add_argument("--api_key", type=str, help="LLM API密钥")
    
    args = parser.parse_args()
    
    # 如果没有指定文件，使用默认的测试文件
    if not args.file:
        test_file = project_root / "data" / "Untitled (1).docx"
        if test_file.exists():
            args.file = str(test_file)
        else:
            logger.error(f"默认测试文件不存在: {test_file}")
            # 列出data目录中的可用文件
            data_dir = project_root / "data"
            if data_dir.exists():
                logger.info("data目录中的可用文件:")
                for f in data_dir.iterdir():
                    if f.suffix in ['.docx', '.pdf', '.txt', '.csv']:
                        logger.info(f"  - {f.name}")
                        if not args.file:
                            args.file = str(f)  # 使用第一个找到的文件
                            break
    
    # 检查文件是否存在
    if not args.file or not os.path.exists(args.file):
        logger.error(f"文件不存在: {args.file}")
        return
    
    # 确保输出目录存在
    output_dir = project_root / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"开始处理文档: {args.file}")
    
    # 初始化转换器
    try:
        converter = DocumentConverter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            llm_provider=args.provider,
            use_ocr=args.use_ocr,
            api_key=args.api_key
        )
    except Exception as e:
        logger.error(f"初始化转换器失败: {e}")
        logger.info("这通常是由于缺少API密钥或依赖库，将继续执行基本的文档加载测试")
        # 继续执行基本测试
        test_basic_loading(args.file, output_dir)
        return
    
    # 处理文档
    try:
        result = converter.process_document(
            file_path=args.file,
            output_formats=args.output_formats
        )
        
        # 输出基本信息
        logger.info(f"文档处理完成，共 {len(result.text_chunks)} 个文本块")
        
        if result.qa_pairs:
            logger.info(f"生成了 {len(result.qa_pairs)} 个问答对")
        
        if result.tables:
            logger.info(f"提取了 {len(result.tables)} 个表格")
        
        # 保存结果
        file_name = os.path.basename(args.file)
        base_name = os.path.splitext(file_name)[0]
        
        # 保存JSON
        if "json" in args.output_formats:
            json_path = output_dir / f"{base_name}_structured.json"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
            logger.info(f"JSON结果已保存到: {json_path}")
        
        # 保存问答对
        if "qa" in args.output_formats and result.qa_pairs:
            qa_path = output_dir / f"{base_name}_qa_pairs.json"
            qa_list = [qa.model_dump() for qa in result.qa_pairs]
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(qa_list, f, ensure_ascii=False, indent=2)
            logger.info(f"问答对已保存到: {qa_path}")
            
        # 保存JSONL
        if "jsonl" in args.output_formats:
            jsonl_path = output_dir / f"{base_name}_chunks.jsonl"
            jsonl_str = converter.convert_to_format(result, "jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(jsonl_str)
            logger.info(f"JSONL结果已保存到: {jsonl_path}")
            
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        logger.info("执行基本文档加载测试...")
        test_basic_loading(args.file, output_dir)

def test_basic_loading(file_path, output_dir):
    """基本文档加载测试（不需要LLM）"""
    try:
        loader = DocumentLoader()
        documents = loader.load_documents([file_path])
        
        logger.info(f"基本加载测试完成:")
        logger.info(f"  - 加载文档数量: {len(documents)}")
        for doc in documents:
            logger.info(f"  - 文档内容长度: {len(doc.page_content)} 字符")
            logger.info(f"  - 文档元数据: {doc.metadata}")
        
        # 保存基本结果
        basic_result = {
            "documents": [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata,
                    "content_length": len(doc.page_content)
                }
                for doc in documents
            ],
            "total_documents": len(documents)
        }
        
        basic_path = output_dir / "basic_loading_test.json"
        with open(basic_path, "w", encoding="utf-8") as f:
            json.dump(basic_result, f, ensure_ascii=False, indent=2)
        logger.info(f"基本加载测试结果已保存到: {basic_path}")
        
    except Exception as e:
        logger.error(f"基本加载测试也失败了: {e}")

if __name__ == "__main__":
    main() 
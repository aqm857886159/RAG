#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目整理脚本：用于清理临时文件、组织文件结构并优化项目组织
"""
import os
import sys
import shutil
import logging
import argparse
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义需要创建的目录
REQUIRED_DIRS = [
    "data",           # 原始数据存储
    "output",         # 转换输出
    "vector_store",   # 向量存储
    "evaluation/data",   # 评估数据
    "evaluation/results",  # 评估结果
    "logs",           # 日志文件
    "tmp",            # 临时文件
    "scripts"         # 脚本文件
]

# 定义临时文件和目录模式
TEMP_PATTERNS = [
    "__pycache__",     # Python缓存
    "*.pyc",           # 编译Python文件
    ".ipynb_checkpoints", # Jupyter临时文件
    "*.tmp",           # 临时文件
    "tmp_*"            # 临时文件前缀
]

# 定义备份文件模式
BACKUP_PATTERNS = [
    "*.backup",        # 备份文件
    "*_backup.*",      # 带_backup后缀的文件
    "* (1).*",         # 副本文件
    "*_copy.*"         # 副本文件
]

# 定义要移动的脚本文件
SCRIPTS_TO_MOVE = {
    "read_docx.py": "scripts/",
    "add_document.py": "scripts/",
    "start.py": "scripts/"
}

def ensure_directories():
    """确保所有必需的目录存在"""
    for directory in REQUIRED_DIRS:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {directory}")
        else:
            logger.info(f"目录已存在: {directory}")

def clean_temp_files():
    """清理临时文件和目录"""
    total_cleaned = 0
    
    # 遍历项目中的所有目录
    for root, dirs, files in os.walk('.'):
        # 跳过.git目录
        if '.git' in root:
            continue
            
        # 清理临时目录
        for temp_dir in [d for d in dirs if any(pattern in d for pattern in TEMP_PATTERNS)]:
            dir_path = os.path.join(root, temp_dir)
            try:
                shutil.rmtree(dir_path)
                logger.info(f"已删除临时目录: {dir_path}")
                total_cleaned += 1
            except Exception as e:
                logger.error(f"无法删除目录 {dir_path}: {str(e)}")
        
        # 清理临时文件
        for file in files:
            if any(file.endswith(pattern.replace('*', '')) for pattern in TEMP_PATTERNS if '*' in pattern):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    logger.info(f"已删除临时文件: {file_path}")
                    total_cleaned += 1
                except Exception as e:
                    logger.error(f"无法删除文件 {file_path}: {str(e)}")
    
    logger.info(f"总共清理了 {total_cleaned} 个临时文件和目录")

def clean_backup_files():
    """清理备份文件"""
    total_cleaned = 0
    
    # 遍历项目中的所有目录
    for root, _, files in os.walk('.'):
        # 跳过.git目录
        if '.git' in root:
            continue
            
        # 清理备份文件
        for file in files:
            if any(file.endswith(pattern.replace('*', '')) for pattern in BACKUP_PATTERNS if '*' in pattern) or \
               any(pattern.replace('*', '') in file for pattern in BACKUP_PATTERNS if '*' in pattern):
                
                # 确认是README备份
                if file == "README.md.backup" or file == "README (1).md":
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"已删除备份文件: {file_path}")
                        total_cleaned += 1
                    except Exception as e:
                        logger.error(f"无法删除文件 {file_path}: {str(e)}")
    
    logger.info(f"总共清理了 {total_cleaned} 个备份文件")

def organize_files():
    """组织项目文件到适当的目录"""
    # 移动原始数据文件到data目录
    data_extensions = ['.pdf', '.docx', '.doc', '.txt', '.csv', '.xlsx', '.xls']
    for file in os.listdir('.'):
        if os.path.isfile(file) and any(file.lower().endswith(ext) for ext in data_extensions):
            if file not in ['README.md', 'requirements.txt', '.env', '.env-example']:
                try:
                    dest = os.path.join('data', file)
                    if not os.path.exists(dest):
                        shutil.move(file, 'data/')
                        logger.info(f"已移动文件到data目录: {file}")
                except Exception as e:
                    logger.error(f"无法移动文件 {file}: {str(e)}")
    
    # 移动测试文件到tests目录
    if not os.path.exists('tests'):
        os.makedirs('tests', exist_ok=True)
        
    for file in os.listdir('.'):
        if os.path.isfile(file) and (file.startswith('test_') or file.endswith('_test.py')):
            try:
                dest = os.path.join('tests', file)
                if not os.path.exists(dest):
                    shutil.move(file, 'tests/')
                    logger.info(f"已移动测试文件到tests目录: {file}")
            except Exception as e:
                logger.error(f"无法移动文件 {file}: {str(e)}")
                
    # 移动脚本文件到scripts目录
    for script, dest_dir in SCRIPTS_TO_MOVE.items():
        if os.path.isfile(script):
            try:
                # 确保目标目录存在
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                
                dest = os.path.join(dest_dir, script)
                if not os.path.exists(dest):
                    shutil.move(script, dest_dir)
                    logger.info(f"已移动脚本文件到{dest_dir}目录: {script}")
            except Exception as e:
                logger.error(f"无法移动文件 {script}: {str(e)}")

def create_env_example():
    """创建或更新.env-example文件"""
    env_example = """# API密钥配置
OPENAI_API_KEY=your_openai_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 模型配置
DEFAULT_LLM_PROVIDER=openai  # 可选: openai, deepseek
OPENAI_MODEL=gpt-3.5-turbo   # 可选: gpt-3.5-turbo, gpt-4
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL=BAAI/bge-large-zh

# 检索配置
TOP_K=5
USE_HYBRID=true
USE_RERANKER=true
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3

# 生成配置
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# 文档处理配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50
USE_OCR=true
QA_PER_CHUNK=3
"""
    
    # 检查是否已存在.env文件
    if not os.path.exists('.env-example'):
        with open('.env-example', 'w', encoding='utf-8') as f:
            f.write(env_example)
        logger.info("已创建.env-example示例文件")
    else:
        logger.info(".env-example文件已存在，跳过创建")

def clean_evaluation_dirs():
    """整理评估目录，确保只有一个标准结构"""
    # 检查是否存在额外的评估目录
    if os.path.exists('evaluation_data') and os.path.exists('evaluation/data'):
        # 移动文件
        for item in os.listdir('evaluation_data'):
            src = os.path.join('evaluation_data', item)
            dst = os.path.join('evaluation/data', item)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                logger.info(f"已复制文件 {src} 到 {dst}")
        # 删除多余目录
        shutil.rmtree('evaluation_data')
        logger.info("已删除多余的evaluation_data目录")
    
    if os.path.exists('evaluation_results') and os.path.exists('evaluation/results'):
        # 移动文件
        for item in os.listdir('evaluation_results'):
            src = os.path.join('evaluation_results', item)
            dst = os.path.join('evaluation/results', item)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                logger.info(f"已复制文件 {src} 到 {dst}")
        # 删除多余目录
        shutil.rmtree('evaluation_results')
        logger.info("已删除多余的evaluation_results目录")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG项目整理工具")
    parser.add_argument("--clean", action="store_true", help="清理临时文件")
    parser.add_argument("--organize", action="store_true", help="组织文件结构")
    parser.add_argument("--backup", action="store_true", help="清理备份文件")
    parser.add_argument("--eval-dirs", action="store_true", help="整理评估目录")
    parser.add_argument("--all", action="store_true", help="执行所有操作")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，则显示帮助
    if not (args.clean or args.organize or args.backup or args.eval_dirs or args.all):
        parser.print_help()
        return
    
    logger.info("开始整理RAG项目...")
    
    # 确保必要的目录存在
    ensure_directories()
    logger.info("已确保必要目录存在")
    
    # 清理临时文件
    if args.clean or args.all:
        clean_temp_files()
        logger.info("临时文件清理完成")
    
    # 清理备份文件
    if args.backup or args.all:
        clean_backup_files()
        logger.info("备份文件清理完成")
    
    # 整理评估目录
    if args.eval_dirs or args.all:
        clean_evaluation_dirs()
        logger.info("评估目录整理完成")
    
    # 组织文件结构
    if args.organize or args.all:
        organize_files()
        logger.info("文件组织完成")
    
    # 创建.env-example
    create_env_example()
    
    logger.info("项目整理完成!")
    
if __name__ == "__main__":
    main() 
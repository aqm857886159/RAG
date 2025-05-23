#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目整理脚本，用于清理和组织项目文件
"""
import os
import shutil
import argparse
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("organize_project.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

# 测试文件目录
TEST_DIR = ROOT_DIR / "tests"

# 定义文件分类
FILE_CATEGORIES = {
    "app_files": [
        "src/app.py",
        "src/app_fixed.py",
        "src/minimal_app.py",
        "app_simple.py"
    ],
    "start_scripts": [
        "start_minimal.py",
        "simple_start.py",
        "robust_start.py",
        "run_app.py",
        "start_app.py"
    ],
    "test_scripts": [
        "test_*.py",
        "ultra_simple_test.py",
        "simple_test.py",
        "test_cmd.py",
        "test_document.txt",
        "test_compatibility.py",
        "test_deepseek.py",
        "test_rag.py",
        "sys_compatibility.py"
    ],
    "backup_files": [
        "*.backup",
        "*.bak",
        "*_backup.*"
    ],
    "temp_files": [
        "*.tmp",
        "*.temp",
        "*.log"
    ]
}

def organize_test_files():
    """整理测试文件到tests目录"""
    logger.info("开始整理测试文件...")
    
    # 确保测试目录存在
    TEST_DIR.mkdir(exist_ok=True)
    
    # 整理测试文件
    test_file_patterns = FILE_CATEGORIES["test_scripts"]
    moved_files = 0
    
    for pattern in test_file_patterns:
        if "*" in pattern:
            # 如果是通配符模式，使用glob
            for test_file in ROOT_DIR.glob(pattern):
                if test_file.is_file() and test_file.parent == ROOT_DIR:
                    dest_file = TEST_DIR / test_file.name
                    try:
                        # 如果目标已存在，使用新名称
                        if dest_file.exists():
                            i = 1
                            while (TEST_DIR / f"{test_file.stem}_{i}{test_file.suffix}").exists():
                                i += 1
                            dest_file = TEST_DIR / f"{test_file.stem}_{i}{test_file.suffix}"
                        
                        # 移动文件
                        shutil.move(str(test_file), str(dest_file))
                        logger.info(f"已移动: {test_file.name} -> {dest_file}")
                        moved_files += 1
                    except Exception as e:
                        logger.error(f"移动文件失败: {test_file}, 错误: {str(e)}")
        else:
            # 如果是具体文件名
            test_file = ROOT_DIR / pattern
            if test_file.is_file():
                dest_file = TEST_DIR / test_file.name
                try:
                    # 如果目标已存在，使用新名称
                    if dest_file.exists():
                        i = 1
                        while (TEST_DIR / f"{test_file.stem}_{i}{test_file.suffix}").exists():
                            i += 1
                        dest_file = TEST_DIR / f"{test_file.stem}_{i}{test_file.suffix}"
                    
                    # 移动文件
                    shutil.move(str(test_file), str(dest_file))
                    logger.info(f"已移动: {test_file.name} -> {dest_file}")
                    moved_files += 1
                except Exception as e:
                    logger.error(f"移动文件失败: {test_file}, 错误: {str(e)}")
    
    logger.info(f"测试文件整理完成，共移动 {moved_files} 个文件")

def clean_temp_files():
    """清理临时文件"""
    logger.info("开始清理临时文件...")
    
    # 整理临时文件
    temp_file_patterns = FILE_CATEGORIES["temp_files"] + FILE_CATEGORIES["backup_files"]
    removed_files = 0
    
    for pattern in temp_file_patterns:
        for temp_file in ROOT_DIR.glob(pattern):
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                    logger.info(f"已删除: {temp_file}")
                    removed_files += 1
                except Exception as e:
                    logger.error(f"删除文件失败: {temp_file}, 错误: {str(e)}")
    
    # 清理临时目录
    tmp_dir = ROOT_DIR / "tmp"
    if tmp_dir.exists() and tmp_dir.is_dir():
        try:
            shutil.rmtree(str(tmp_dir))
            logger.info("已删除临时目录: tmp/")
            removed_files += 1
        except Exception as e:
            logger.error(f"删除临时目录失败: tmp/, 错误: {str(e)}")
    
    logger.info(f"临时文件清理完成，共删除 {removed_files} 个文件/目录")

def organize_app_files():
    """整理应用文件"""
    logger.info("开始整理应用文件...")
    
    # 确保src目录存在
    src_dir = ROOT_DIR / "src"
    src_dir.mkdir(exist_ok=True)
    
    # 移动散落的应用文件到src目录
    app_files = [f for f in FILE_CATEGORIES["app_files"] if not f.startswith("src/")]
    moved_files = 0
    
    for app_file in app_files:
        file_path = ROOT_DIR / app_file
        if file_path.is_file():
            dest_file = src_dir / file_path.name
            try:
                # 如果目标已存在，使用新名称
                if dest_file.exists():
                    i = 1
                    while (src_dir / f"{file_path.stem}_{i}{file_path.suffix}").exists():
                        i += 1
                    dest_file = src_dir / f"{file_path.stem}_{i}{file_path.suffix}"
                
                # 移动文件
                shutil.move(str(file_path), str(dest_file))
                logger.info(f"已移动: {file_path.name} -> {dest_file}")
                moved_files += 1
            except Exception as e:
                logger.error(f"移动文件失败: {file_path}, 错误: {str(e)}")
    
    logger.info(f"应用文件整理完成，共移动 {moved_files} 个文件")

def create_backup():
    """创建项目备份"""
    logger.info("开始创建项目备份...")
    
    # 创建备份目录
    backup_dir = ROOT_DIR / "backup"
    timestamp = backup_dir.absolute().as_posix().replace('/', '_').replace('\\', '_').replace(':', '')
    backup_dir = ROOT_DIR / f"backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    
    # 复制核心文件
    core_files = []
    for file_category in ["app_files", "start_scripts"]:
        core_files.extend(FILE_CATEGORIES[file_category])
    
    copied_files = 0
    for file_path in core_files:
        src_file = ROOT_DIR / file_path
        if src_file.exists() and src_file.is_file():
            dest_file = backup_dir / src_file.name
            try:
                shutil.copy2(str(src_file), str(dest_file))
                logger.info(f"已备份: {src_file.name} -> {dest_file}")
                copied_files += 1
            except Exception as e:
                logger.error(f"备份文件失败: {src_file}, 错误: {str(e)}")
    
    # 复制src目录
    src_dir = ROOT_DIR / "src"
    if src_dir.exists() and src_dir.is_dir():
        dest_dir = backup_dir / "src"
        try:
            shutil.copytree(str(src_dir), str(dest_dir))
            logger.info(f"已备份: src/ -> {dest_dir}")
            copied_files += 1
        except Exception as e:
            logger.error(f"备份目录失败: src/, 错误: {str(e)}")
    
    logger.info(f"项目备份完成，共备份 {copied_files} 个文件/目录")
    logger.info(f"备份目录: {backup_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="项目整理脚本")
    parser.add_argument("--test", action="store_true", help="整理测试文件")
    parser.add_argument("--temp", action="store_true", help="清理临时文件")
    parser.add_argument("--app", action="store_true", help="整理应用文件")
    parser.add_argument("--backup", action="store_true", help="创建项目备份")
    parser.add_argument("--all", action="store_true", help="执行所有整理操作")
    
    args = parser.parse_args()
    
    # 如果没有指定任何操作，显示帮助信息
    if not (args.test or args.temp or args.app or args.backup or args.all):
        parser.print_help()
        return
    
    # 执行整理操作
    if args.all or args.backup:
        create_backup()
    
    if args.all or args.test:
        organize_test_files()
    
    if args.all or args.app:
        organize_app_files()
    
    if args.all or args.temp:
        clean_temp_files()
    
    logger.info("项目整理完成")

if __name__ == "__main__":
    main() 
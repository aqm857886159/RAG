#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的脚本来整理测试文件
"""
import os
import shutil
import sys
from pathlib import Path

print("开始执行整理脚本...")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")

try:
    # 测试文件列表
    TEST_FILES = [
        "test_rag.py",
        "test_deepseek.py",
        "ultra_simple_test.py",
        "simple_test.py",
        "test_compatibility.py",
        "test_cmd.py",
        "test_document.txt",
        "sys_compatibility.py"
    ]
    
    # 项目根目录
    ROOT_DIR = Path(os.getcwd())
    print(f"项目根目录: {ROOT_DIR}")
    
    # 测试目录
    TEST_DIR = ROOT_DIR / "tests"
    TEST_DIR.mkdir(exist_ok=True)
    print(f"测试目录: {TEST_DIR}")
    
    # 移动测试文件
    moved_files = 0
    for test_file in TEST_FILES:
        src_path = ROOT_DIR / test_file
        if src_path.exists():
            dst_path = TEST_DIR / test_file
            try:
                shutil.copy2(str(src_path), str(dst_path))
                print(f"已复制: {test_file} -> {dst_path}")
                moved_files += 1
            except Exception as e:
                print(f"复制文件失败: {test_file}, 错误: {str(e)}")
        else:
            print(f"文件不存在: {src_path}")
    
    print(f"测试文件整理完成，共复制 {moved_files} 个文件")
    
    # 创建备份
    BACKUP_DIR = ROOT_DIR / "backup"
    BACKUP_DIR.mkdir(exist_ok=True)
    print(f"备份目录: {BACKUP_DIR}")
    
    # 复制重要文件到备份
    important_files = [
        "start_minimal.py",
        "simple_start.py",
        "robust_start.py",
        "run_app.py"
    ]
    
    for file in important_files:
        src_file = ROOT_DIR / file
        if src_file.exists():
            dst_file = BACKUP_DIR / file
            try:
                shutil.copy2(str(src_file), str(dst_file))
                print(f"已备份: {file} -> {dst_file}")
            except Exception as e:
                print(f"备份文件失败: {file}, 错误: {str(e)}")
    
    # 复制src目录到备份
    SRC_DIR = ROOT_DIR / "src"
    BACKUP_SRC_DIR = BACKUP_DIR / "src"
    
    if SRC_DIR.exists():
        try:
            if not BACKUP_SRC_DIR.exists():
                BACKUP_SRC_DIR.mkdir(exist_ok=True)
            
            # 复制src目录中的文件
            for item in SRC_DIR.glob("*"):
                if item.is_file():
                    shutil.copy2(str(item), str(BACKUP_SRC_DIR / item.name))
                    print(f"已备份: {item.name} -> {BACKUP_SRC_DIR / item.name}")
            
            print(f"已备份src目录中的文件到: {BACKUP_SRC_DIR}")
        except Exception as e:
            print(f"备份src目录失败: {str(e)}")
    else:
        print(f"src目录不存在: {SRC_DIR}")
    
    print("整理完成!")
except Exception as e:
    print(f"整理过程中出错: {str(e)}")
    import traceback
    traceback.print_exc() 
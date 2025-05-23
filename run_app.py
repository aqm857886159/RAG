#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动脚本，用于运行Streamlit应用
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入系统兼容性模块，确保在导入其他模块之前进行系统兼容性修复
try:
    from src.utils.sys_compatibility import patch_unix_modules
    # 显式调用一次，确保补丁已应用
    patch_unix_modules()
    logging.info("已应用系统兼容性补丁")
except Exception as e:
    logging.warning(f"应用系统兼容性补丁失败: {str(e)}")

import subprocess
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保必要的目录存在
def ensure_directories():
    """确保必要的目录存在"""
    required_dirs = [
        "data",
        "vector_store",
        "evaluation/data",
        "evaluation/results",
        "output"
    ]
    
    for directory in required_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {directory}")

def run_project_cleanup(organize=False):
    """运行项目整理脚本"""
    cleanup_script = Path("clean_project.py")
    if cleanup_script.exists():
        cmd = ["python", str(cleanup_script)]
        
        if organize:
            cmd.append("--organize")
        else:
            cmd.append("--clean")
        
        try:
            logger.info("正在运行项目整理脚本...")
            subprocess.run(cmd, check=True)
            logger.info("项目整理完成")
        except subprocess.CalledProcessError as e:
            logger.warning(f"项目整理脚本运行失败: {str(e)}")
    else:
        logger.warning("未找到项目整理脚本: clean_project.py")

def main():
    """主函数"""
    # 加载环境变量
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv()
        logger.info("已加载.env环境变量")
    else:
        logger.warning("未找到.env文件，将使用默认配置")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="RAG应用启动工具")
    parser.add_argument("--port", type=int, default=8501, help="应用服务端口")
    parser.add_argument("--browser", action="store_true", help="自动打开浏览器")
    parser.add_argument("--mode", choices=["rag", "converter"], help="直接启动特定模式")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--clean", action="store_true", help="启动前运行项目整理(清理临时文件)")
    parser.add_argument("--organize", action="store_true", help="启动前运行项目整理(组织文件结构)")
    
    args = parser.parse_args()
    
    # 如果指定了整理项目，运行项目整理脚本
    if args.clean:
        run_project_cleanup(organize=False)
    elif args.organize:
        run_project_cleanup(organize=True)
    
    # 确保必要的目录存在
    ensure_directories()
    
    # 设置调试模式
    if args.debug:
        os.environ["DEBUG"] = "true"
        logger.info("已启用调试模式")
    
    # 设置默认模式
    if args.mode:
        os.environ["DEFAULT_MODE"] = args.mode
        logger.info(f"默认模式设置为: {args.mode}")
    
    # 构建Streamlit命令行参数
    streamlit_args = ["streamlit", "run", "src/app.py"]
    
    # 添加端口
    streamlit_args.extend(["--server.port", str(args.port)])
    
    # 添加浏览器选项
    if not args.browser:
        streamlit_args.extend(["--server.headless", "true"])
    
    # 启动Streamlit应用
    logger.info(f"启动Streamlit应用，端口: {args.port}")
    sys.argv = streamlit_args
    
    import streamlit.web.cli as stcli
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 
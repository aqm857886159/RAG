#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
健壮的启动脚本，处理可能的错误并提供备选方案
"""
import os
import sys
import logging
import time
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("start_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 尝试修复常见问题
def fix_common_issues():
    """尝试修复常见问题"""
    issues_fixed = 0
    
    # 确保必要的目录存在
    for dir_name in ["data", "vector_store", "evaluation_data", "evaluation_results"]:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {dir_name}")
            issues_fixed += 1
    
    return issues_fixed

def check_app_file():
    """检查应用文件状态"""
    app_py = Path("src/app.py")
    app_fixed_py = Path("src/app_fixed.py")
    
    if app_fixed_py.exists():
        logger.info("发现app_fixed.py，将使用该文件")
        return "src/app_fixed.py"
    elif app_py.exists():
        logger.info("将使用src/app.py")
        return "src/app.py"
    else:
        logger.error("未找到应用文件(app.py或app_fixed.py)")
        return None

def main():
    """主函数"""
    print("准备启动RAG应用...")
    
    # 修复常见问题
    issues_fixed = fix_common_issues()
    if issues_fixed > 0:
        print(f"已修复 {issues_fixed} 个常见问题")
    
    # 检查应用文件
    app_file = check_app_file()
    if not app_file:
        print("错误: 未找到应用文件，无法启动")
        return
    
    # 设置端口和其他参数
    port = 8510  # 使用不同端口避免冲突
    
    # 尝试导入和使用streamlit
    try:
        import streamlit.web.cli as stcli
        
        # 设置Streamlit命令行参数
        sys.argv = [
            "streamlit", 
            "run", 
            app_file,
            "--server.port", 
            str(port)
        ]
        
        # 启动Streamlit应用
        print(f"正在启动应用，使用文件: {app_file}，端口: {port}")
        print("应用启动后，请在浏览器中访问: http://localhost:8510")
        print("如果浏览器没有自动打开，请手动打开上述链接")
        print("按Ctrl+C终止应用")
        
        # 等待一秒，让用户看到上述信息
        time.sleep(1)
        
        # 启动应用
        sys.exit(stcli.main())
    
    except Exception as e:
        logger.error(f"启动应用时出错: {str(e)}")
        print(f"错误: {str(e)}")
        print("请检查日志文件start_app.log获取更多信息")
        
        # 如果是ModuleNotFoundError，提供安装建议
        if isinstance(e, ModuleNotFoundError):
            missing_module = str(e).split("'")[1]
            print(f"\n可能缺少必要的模块: {missing_module}")
            print(f"请尝试运行: pip install {missing_module}")
            print("或者运行: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 
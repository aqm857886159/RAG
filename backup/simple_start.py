#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版启动脚本，使用app_fixed.py
"""
import os
import sys
import streamlit.web.cli as stcli

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    """主函数"""
    print("正在启动应用，使用app_fixed.py...")
    
    # 设置Streamlit命令行参数
    port = 8510  # 使用不同端口避免冲突
    sys.argv = [
        "streamlit", 
        "run", 
        "src/app_fixed.py",
        "--server.port", 
        str(port)
    ]
    
    # 启动Streamlit应用
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
启动最小化RAG应用
"""
import os
import sys
import streamlit.web.cli as stcli

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    """主函数"""
    print("正在启动简化版RAG应用...")
    
    # 设置Streamlit命令行参数
    port = 8511
    sys.argv = [
        "streamlit", 
        "run", 
        "src/minimal_app.py",
        "--server.port", 
        str(port)
    ]
    
    # 启动Streamlit应用
    print(f"应用将在端口 {port} 上运行")
    print(f"请访问: http://localhost:{port}")
    print("按Ctrl+C终止应用")
    
    # 启动应用
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 
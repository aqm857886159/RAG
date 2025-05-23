#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的启动脚本，专注于加载兼容性模块
"""

# 最先导入系统兼容性模块，在任何其他导入之前
print("正在应用系统兼容性补丁...")
import sys_compatibility

# 现在导入标准库
import os
import sys
import streamlit.web.cli as stcli

def main():
    """主函数"""
    print("启动Streamlit应用...")
    
    # 设置命令行参数
    sys.argv = ["streamlit", "run", "src/app.py", "--server.headless", "true"]
    
    # 启动Streamlit应用
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 
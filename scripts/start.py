"""
启动脚本
"""
import os
import sys
import subprocess

# 直接运行streamlit命令
os.environ["PYTHONPATH"] = os.getcwd()
subprocess.run(["streamlit", "run", "src/app.py"]) 
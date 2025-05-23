"""
启动修复版RAG应用的脚本
"""
import os
import sys
import subprocess
from pathlib import Path
import logging
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("startup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """启动修复版RAG应用"""
    parser = argparse.ArgumentParser(description="启动RAG应用")
    parser.add_argument("--port", type=int, default=8502, help="应用端口号")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="主机地址")
    parser.add_argument("--app", type=str, default="enhanced_fixed_app.py", 
                        help="要运行的应用文件名(enhanced_fixed_app.py或app.py)")
    args = parser.parse_args()
    
    logger.info(f"正在启动RAG应用: {args.app}...")
    
    # 获取当前脚本路径
    script_dir = Path(__file__).parent.absolute()
    
    # 确保环境变量中包含必要的路径
    sys.path.insert(0, str(script_dir))
    
    # 构建命令
    app_path = script_dir / "src" / args.app
    if not app_path.exists():
        logger.error(f"应用文件不存在: {app_path}")
        print(f"错误: 应用文件不存在: {app_path}")
        print(f"可用的应用文件:")
        for app_file in (script_dir / "src").glob("*.py"):
            if "app" in app_file.name:
                print(f"  - {app_file.name}")
        sys.exit(1)
    
    cmd = [sys.executable, "-m", "streamlit", "run", 
           str(app_path),
           f"--server.port={args.port}", 
           f"--server.address={args.host}"]
    
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        print(f"正在启动应用: {args.app}")
        print(f"访问地址: http://{args.host}:{args.port}")
        # 运行应用
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("应用已手动停止")
        print("\n应用已停止")
    except Exception as e:
        logger.error(f"启动应用时发生错误: {str(e)}")
        print(f"启动应用时发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
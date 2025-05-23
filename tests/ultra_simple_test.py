"""
极度简化的测试脚本，只测试环境和基本依赖
"""
import os
import sys
import platform
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境"""
    print("=== 环境信息 ===")
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"平台: {sys.platform}")
    print("=================")

def test_imports():
    """测试基本依赖导入"""
    imports = [
        "langchain", 
        "langchain_community",
        "langchain_openai", 
        "numpy",
        "pandas",
        "transformers",
        "torch"
    ]
    
    print("=== 依赖测试 ===")
    for lib in imports:
        try:
            __import__(lib)
            print(f"✓ {lib}")
        except ImportError:
            print(f"✗ {lib} - 导入失败")
    print("=================")

def main():
    """主函数"""
    check_environment()
    test_imports()
    
    # 测试虚拟pwd模块
    try:
        import pwd
        user = pwd.getpwuid(os.getuid()).pw_name
        print(f"虚拟pwd模块测试成功，当前用户: {user}")
    except Exception as e:
        print(f"虚拟pwd模块测试失败: {str(e)}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 
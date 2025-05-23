"""
应用程序启动脚本
自动应用系统兼容性修复和向量存储检查
"""
import os
import sys
import logging
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    print("📚 初始化RAG系统...")
    
    # 第一步：应用系统兼容性补丁
    print("🔧 应用系统兼容性补丁...")
    try:
        from src.utils.sys_compatibility import patch_unix_modules
        patch_unix_modules()
        print("✅ 系统兼容性补丁已应用")
    except Exception as e:
        print(f"⚠️ 应用系统兼容性补丁失败: {str(e)}")
        
        # 第二种尝试
        try:
            from utils.sys_compatibility import patch_unix_modules
            patch_unix_modules()
            print("✅ 通过备选路径应用系统兼容性补丁")
        except Exception as inner_e:
            print(f"⚠️ 通过备选路径应用兼容性补丁失败: {str(inner_e)}")
            print("⚠️ 继续运行，但可能在Windows系统上遇到兼容性问题")
    
    # 第二步：检查向量存储目录
    print("\n🔍 检查向量存储目录...")
    try:
        from src.utils.check_vector_store import check_and_fix_vector_store
        results = check_and_fix_vector_store()
        
        if results["overall_status"]:
            print("✅ 向量存储目录检查通过")
        else:
            print("⚠️ 向量存储目录存在问题，已尝试修复")
            if results["fixes_applied"]:
                for fix in results["fixes_applied"]:
                    print(f"  - {fix}")
    except Exception as e:
        print(f"⚠️ 检查向量存储目录失败: {str(e)}")
        print("⚠️ 继续运行，但可能无法正常使用向量存储功能")
    
    # 第三步：启动应用程序
    print("\n🚀 启动应用程序...")
    try:
        import streamlit.web.cli as stcli
        
        # 构建命令行参数
        sys.argv = ["streamlit", "run", "src/app.py", "--server.headless", "false"]
        
        # 启动Streamlit应用
        sys.exit(stcli.main())
    except Exception as e:
        print(f"❌ 启动应用程序失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
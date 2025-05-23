"""
向量存储检查和修复工具
用于自动检查向量存储目录是否正确，并尝试修复常见问题
"""
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入配置
from config.config import VECTOR_STORE_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_directory_permissions(directory: str) -> Tuple[bool, str]:
    """
    检查目录权限
    
    Args:
        directory: 要检查的目录路径
        
    Returns:
        (是否通过, 消息)
    """
    try:
        # 检查目录是否存在
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
        
        # 检查读权限
        if not os.access(directory, os.R_OK):
            return False, f"目录 {directory} 不可读"
            
        # 检查写权限
        if not os.access(directory, os.W_OK):
            return False, f"目录 {directory} 不可写"
            
        # 创建测试文件检查实际写入
        test_file = os.path.join(directory, ".permission_test")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            return False, f"无法在目录 {directory} 中写入文件: {str(e)}"
            
        return True, f"目录 {directory} 权限正常"
        
    except Exception as e:
        return False, f"检查目录 {directory} 权限时出错: {str(e)}"

def check_faiss_store(faiss_dir: str) -> Tuple[bool, str]:
    """
    检查FAISS向量存储
    
    Args:
        faiss_dir: FAISS向量存储目录
        
    Returns:
        (是否通过, 消息)
    """
    try:
        # 检查目录权限
        dir_ok, dir_msg = check_directory_permissions(faiss_dir)
        if not dir_ok:
            return dir_ok, dir_msg
            
        # 检查索引文件
        index_file = os.path.join(faiss_dir, "index.faiss")
        if not os.path.exists(index_file):
            return False, f"FAISS索引文件不存在: {index_file}"
            
        # 检查索引文件大小
        if os.path.getsize(index_file) == 0:
            return False, f"FAISS索引文件为空: {index_file}"
            
        # 检查文档存储文件
        docstore_file = os.path.join(faiss_dir, "index.pkl")
        if not os.path.exists(docstore_file):
            return False, f"FAISS文档存储文件不存在: {docstore_file}"
            
        return True, "FAISS向量存储正常"
        
    except Exception as e:
        return False, f"检查FAISS向量存储时出错: {str(e)}"

def check_chroma_store(chroma_dir: str) -> Tuple[bool, str]:
    """
    检查Chroma向量存储
    
    Args:
        chroma_dir: Chroma向量存储目录
        
    Returns:
        (是否通过, 消息)
    """
    try:
        # 检查目录权限
        dir_ok, dir_msg = check_directory_permissions(chroma_dir)
        if not dir_ok:
            return dir_ok, dir_msg
            
        # 检查数据库文件
        db_file = os.path.join(chroma_dir, "chroma.sqlite3")
        if not os.path.exists(db_file):
            return False, f"Chroma数据库文件不存在: {db_file}"
            
        # 检查数据库文件大小
        if os.path.getsize(db_file) == 0:
            return False, f"Chroma数据库文件为空: {db_file}"
            
        return True, "Chroma向量存储正常"
        
    except Exception as e:
        return False, f"检查Chroma向量存储时出错: {str(e)}"

def fix_directory_permissions(directory: str) -> Tuple[bool, str]:
    """
    修复目录权限问题
    
    Args:
        directory: 要修复的目录
        
    Returns:
        (是否成功, 消息)
    """
    try:
        if not os.path.exists(directory):
            # 创建目录
            os.makedirs(directory, exist_ok=True)
            logger.info(f"创建目录: {directory}")
            
        # 尝试使用备用目录
        backup_dir = os.path.join(os.path.expanduser("~"), "rag_vector_store")
        
        # 确保备用目录存在
        os.makedirs(backup_dir, exist_ok=True)
        
        # 创建符号链接或目录映射
        try:
            # 尝试Windows特有的目录映射(junction)
            if os.name == 'nt':
                # 移除原目录和目标目录(如果已存在)
                if os.path.exists(directory):
                    try:
                        shutil.rmtree(directory)
                    except:
                        logger.warning(f"无法删除原目录: {directory}")
                
                # 创建目录连接
                os.system(f'mklink /J "{directory}" "{backup_dir}"')
                logger.info(f"创建目录映射: {directory} -> {backup_dir}")
            else:
                # 在Unix系统上创建符号链接
                if os.path.exists(directory):
                    os.remove(directory)
                os.symlink(backup_dir, directory)
                logger.info(f"创建符号链接: {directory} -> {backup_dir}")
                
            return True, f"已创建目录映射或符号链接: {directory} -> {backup_dir}"
        except Exception as e:
            logger.error(f"创建目录映射失败: {str(e)}")
            
            # 如果无法创建映射，使用配置文件修改
            try:
                from config import config
                setattr(config, "VECTOR_STORE_DIR", backup_dir)
                logger.info(f"已修改配置中的向量存储目录为: {backup_dir}")
                return True, f"已修改配置中的向量存储目录为: {backup_dir}"
            except Exception as config_e:
                logger.error(f"修改配置失败: {str(config_e)}")
                return False, f"无法修复目录权限问题"
    
    except Exception as e:
        return False, f"修复目录权限时出错: {str(e)}"

def check_and_fix_vector_store(vector_store_dir: str = VECTOR_STORE_DIR) -> Dict[str, Any]:
    """
    检查并修复向量存储
    
    Args:
        vector_store_dir: 向量存储目录
        
    Returns:
        结果字典
    """
    results = {
        "main_dir": {"status": False, "message": ""},
        "faiss": {"status": False, "message": ""},
        "chroma": {"status": False, "message": ""},
        "fixes_applied": [],
        "overall_status": False
    }
    
    # 规范化路径
    vector_store_dir = os.path.normpath(vector_store_dir)
    
    # 检查主目录
    main_ok, main_msg = check_directory_permissions(vector_store_dir)
    results["main_dir"]["status"] = main_ok
    results["main_dir"]["message"] = main_msg
    
    # 如果主目录有问题，尝试修复
    if not main_ok:
        fix_ok, fix_msg = fix_directory_permissions(vector_store_dir)
        if fix_ok:
            results["fixes_applied"].append(f"修复主目录: {fix_msg}")
            # 重新检查
            main_ok, main_msg = check_directory_permissions(vector_store_dir)
            results["main_dir"]["status"] = main_ok
            results["main_dir"]["message"] = f"{main_msg} (已修复)"
        else:
            results["fixes_applied"].append(f"修复主目录失败: {fix_msg}")
    
    # 检查默认索引目录
    default_dir = os.path.join(vector_store_dir, "default")
    os.makedirs(default_dir, exist_ok=True)
    
    # 检查FAISS
    faiss_dir = default_dir
    faiss_ok, faiss_msg = check_faiss_store(faiss_dir)
    results["faiss"]["status"] = faiss_ok
    results["faiss"]["message"] = faiss_msg
    
    # 检查Chroma
    chroma_dir = default_dir
    chroma_ok, chroma_msg = check_chroma_store(chroma_dir)
    results["chroma"]["status"] = chroma_ok
    results["chroma"]["message"] = chroma_msg
    
    # 设置总体状态
    results["overall_status"] = main_ok and (faiss_ok or chroma_ok)
    
    return results

def main():
    """主函数"""
    # 显示检查结果
    print("检查向量存储目录...")
    results = check_and_fix_vector_store()
    
    print(f"\n向量存储目录: {VECTOR_STORE_DIR}")
    print(f"主目录状态: {'✅' if results['main_dir']['status'] else '❌'} {results['main_dir']['message']}")
    print(f"FAISS状态: {'✅' if results['faiss']['status'] else '❌'} {results['faiss']['message']}")
    print(f"Chroma状态: {'✅' if results['chroma']['status'] else '❌'} {results['chroma']['message']}")
    
    if results["fixes_applied"]:
        print("\n应用的修复:")
        for fix in results["fixes_applied"]:
            print(f"- {fix}")
    
    print(f"\n总体状态: {'✅ 正常' if results['overall_status'] else '❌ 存在问题'}")
    
    if not results["overall_status"]:
        print("\n建议操作:")
        print("1. 确保应用程序有足够权限访问向量存储目录")
        print("2. 尝试手动创建索引目录并设置适当权限")
        print("3. 考虑更改配置文件中的向量存储目录位置")
        print("4. 尝试在管理员权限下运行应用程序")
    
if __name__ == "__main__":
    main() 
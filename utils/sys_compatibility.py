"""
系统兼容性模块（项目根目录版本），解决Windows和Unix/Linux系统之间的兼容性问题
"""
import os
import sys
import types
import platform
import logging

logger = logging.getLogger(__name__)

def patch_pwd_module():
    """
    为Windows系统创建一个虚拟的pwd模块
    在Unix/Linux系统上，这个函数不做任何事情
    """
    if platform.system() == "Windows" and "pwd" not in sys.modules:
        print("检测到Windows系统，创建虚拟pwd模块")
        
        # 创建模拟的pwd模块
        pwd_module = types.ModuleType("pwd")
        
        # 添加必要的函数和属性
        def getpwuid(uid):
            """模拟getpwuid函数，返回一个包含pw_name的字典"""
            return {"pw_name": os.environ.get("USERNAME", "user")}
        
        pwd_module.getpwuid = getpwuid
        
        # 将模块添加到sys.modules中
        sys.modules["pwd"] = pwd_module
        
        print("已成功创建虚拟pwd模块")
        return True
    return False

def patch_unix_modules():
    """
    为Windows系统添加所有需要的Unix特有模块的兼容层
    """
    patched = []
    
    # 添加pwd模块补丁
    if patch_pwd_module():
        patched.append("pwd")
    
    # 未来可以在这里添加其他Unix特有模块的补丁
    # 例如: grp, fcntl等
    
    if patched:
        print(f"已为Windows系统添加以下模块的兼容层: {', '.join(patched)}")

# 在导入时自动应用补丁
patch_unix_modules() 
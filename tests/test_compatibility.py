"""
简单的兼容性测试脚本，用于测试Windows环境下的RAG系统组件
"""
import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_compatibility.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入sys_compatibility模块确保它在所有导入之前运行
import sys_compatibility

def test_llm():
    """测试LLM模块"""
    logger.info("测试LLM模块...")
    try:
        from src.llm import get_llm, get_available_llm_providers
        
        # 获取可用提供商
        providers = get_available_llm_providers()
        logger.info(f"可用的LLM提供商: {providers}")
        
        # 尝试获取LLM
        llm = get_llm()
        logger.info(f"成功创建LLM: {llm}")
        
        return True
    except Exception as e:
        logger.error(f"LLM模块测试失败: {str(e)}")
        return False

def test_vector_store():
    """测试向量存储模块"""
    logger.info("测试向量存储模块...")
    try:
        from src.vectorizer import Vectorizer
        
        # 尝试创建向量化器
        vectorizer = Vectorizer()
        logger.info(f"成功创建向量化器: {vectorizer}")
        
        return True
    except Exception as e:
        logger.error(f"向量存储模块测试失败: {str(e)}")
        return False

def test_generator():
    """测试生成器模块"""
    logger.info("测试生成器模块...")
    try:
        from src.generator import RAGGenerator, SimpleLLMGenerator
        from src.llm import get_llm
        from src.vectorizer import Vectorizer
        
        # 尝试创建简单生成器
        llm = get_llm()
        simple_generator = SimpleLLMGenerator(llm=llm)
        logger.info(f"成功创建简单生成器: {simple_generator}")
        
        # 尝试创建RAG生成器
        vectorizer = Vectorizer()
        rag_generator = RAGGenerator(llm=llm, vectorizer=vectorizer)
        logger.info(f"成功创建RAG生成器: {rag_generator}")
        
        return True
    except Exception as e:
        logger.error(f"生成器模块测试失败: {str(e)}")
        return False

def main():
    """运行所有测试"""
    logger.info("开始兼容性测试...")
    
    # 检查Python版本和操作系统
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"操作系统: {sys.platform}")
    
    # 运行测试
    tests = [
        ("LLM模块", test_llm),
        ("向量存储模块", test_vector_store),
        ("生成器模块", test_generator)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"运行测试: {name}")
        success = test_func()
        results.append((name, success))
        logger.info(f"测试结果: {name} - {'成功' if success else '失败'}")
        
    # 打印测试结果摘要
    logger.info("==== 测试结果摘要 ====")
    for name, success in results:
        logger.info(f"{name}: {'✓' if success else '✗'}")
    
    # 检查是否所有测试都通过
    if all(success for _, success in results):
        logger.info("所有测试通过！")
    else:
        logger.info("有测试失败，请查看日志获取详细信息。")

if __name__ == "__main__":
    main() 
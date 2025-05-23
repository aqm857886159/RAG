"""
测试DeepSeek LLM连接和功能
"""
import logging
import os
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_deepseek_connection():
    """测试DeepSeek连接"""
    try:
        from src.llm import get_llm, get_available_llm_providers
        from langchain.schema.messages import HumanMessage
        
        # 检查可用提供商
        providers = get_available_llm_providers()
        logger.info(f"可用的LLM提供商: {providers}")
        
        if not providers.get("deepseek", False):
            logger.error("DeepSeek API密钥未配置或无效")
            return False
            
        # 获取DeepSeek LLM实例
        logger.info("尝试创建DeepSeek LLM实例...")
        llm = get_llm(provider="deepseek")
        logger.info(f"成功创建DeepSeek LLM: {type(llm).__name__}")
        
        # 测试一个简单问题
        question = "你好，请用一句话介绍一下自己"
        logger.info(f"向DeepSeek提问: {question}")
        
        # 使用正确的消息类型
        messages = [HumanMessage(content=question)]
        response = llm.invoke(messages)
        
        answer = response.content
        logger.info(f"DeepSeek回答: {answer}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试DeepSeek连接失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    logger.info("开始测试DeepSeek LLM...")
    
    # 测试DeepSeek连接
    success = test_deepseek_connection()
    
    if success:
        logger.info("DeepSeek LLM测试成功！")
        print("✅ DeepSeek LLM测试成功！")
    else:
        logger.error("DeepSeek LLM测试失败！")
        print("❌ DeepSeek LLM测试失败！请检查日志获取详细信息。")

if __name__ == "__main__":
    main() 
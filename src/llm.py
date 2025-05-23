"""
LLM模块，用于初始化和管理大语言模型
"""
import os
import logging
import json
from typing import Optional, Dict, Any, Union

from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from models.deepseek_llm import DeepSeekLLM
from config.config import (
    OPENAI_API_KEY, OPENAI_MODEL, 
    DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_BASE,
    DEFAULT_LLM_PROVIDER, LLM_TEMPERATURE, LLM_MAX_TOKENS
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMFactory:
    """LLM工厂类，负责创建不同的LLM实例"""
    
    @staticmethod
    def create_openai_llm(
        model_name: str = OPENAI_MODEL,
        api_key: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        **kwargs
    ) -> ChatOpenAI:
        """
        创建OpenAI LLM实例
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大Token数
            
        Returns:
            ChatOpenAI实例
        """
        # 获取API密钥
        api_key = api_key or os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
        
        if not api_key:
            logger.error("未提供OpenAI API密钥")
            raise ValueError("未提供OpenAI API密钥，请检查环境变量或配置文件")
            
        logger.info(f"创建OpenAI LLM: {model_name}")
        
        # 创建OpenAI LLM
        try:
            return ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"创建OpenAI LLM失败: {str(e)}")
            raise
    
    @staticmethod
    def create_deepseek_llm(
        model_name: str = DEEPSEEK_MODEL,
        api_key: Optional[str] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        **kwargs
    ) -> DeepSeekLLM:
        """
        创建DeepSeek LLM实例
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大Token数
            
        Returns:
            DeepSeekLLM实例
        """
        # 获取API密钥
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
        
        if not api_key:
            logger.error("未提供DeepSeek API密钥")
            raise ValueError("未提供DeepSeek API密钥，请检查环境变量或配置文件")
            
        logger.info(f"创建DeepSeek LLM: {model_name}")
        
        # 创建DeepSeek LLM
        try:
            return DeepSeekLLM(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"创建DeepSeek LLM失败: {str(e)}")
            raise

def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    **kwargs
) -> BaseChatModel:
    """
    获取LLM实例
    
    Args:
        provider: 模型提供商，支持 openai 和 deepseek
        model_name: 模型名称
        api_key: API密钥
        temperature: 温度参数
        max_tokens: 最大Token数
        
    Returns:
        LLM实例
    """
    # 确定提供商
    provider = provider or DEFAULT_LLM_PROVIDER
    provider = provider.lower()
    
    # 创建LLM
    try:
        if provider == "openai":
            return LLMFactory.create_openai_llm(
                model_name=model_name or OPENAI_MODEL,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        elif provider == "deepseek":
            return LLMFactory.create_deepseek_llm(
                model_name=model_name or DEEPSEEK_MODEL,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        else:
            logger.error(f"不支持的模型提供商: {provider}")
            raise ValueError(f"不支持的模型提供商: {provider}，当前支持: openai, deepseek")
    except Exception as e:
        logger.error(f"获取LLM实例失败: {str(e)}")
        
        # 尝试使用备选提供商
        backup_provider = "openai" if provider != "openai" else "deepseek"
        logger.info(f"尝试使用备选提供商: {backup_provider}")
        
        try:
            if backup_provider == "openai":
                return LLMFactory.create_openai_llm(
                    model_name=OPENAI_MODEL,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            else:
                return LLMFactory.create_deepseek_llm(
                    model_name=DEEPSEEK_MODEL,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
        except Exception as inner_e:
            logger.error(f"使用备选提供商失败: {str(inner_e)}")
            raise ValueError(f"获取LLM实例失败: {str(e)}，备选提供商也失败: {str(inner_e)}")

def get_available_llm_providers() -> Dict[str, bool]:
    """
    获取当前可用的LLM提供商
    
    Returns:
        字典，提供商名称为键，可用状态为值
    """
    providers = {
        "openai": False,
        "deepseek": False
    }
    
    # 检查OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if openai_api_key:
        providers["openai"] = True
    
    # 检查DeepSeek
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
    if deepseek_api_key:
        providers["deepseek"] = True
    
    return providers 
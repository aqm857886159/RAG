"""
DeepSeek LLM 接口，用于在LangChain中使用DeepSeek API
"""
import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Union, Mapping, Tuple
import time

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.pydantic_v1 import root_validator, BaseModel
from langchain.schema import LLMResult

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepSeekLLM(LLM):
    """
    DeepSeek语言模型的LangChain接口
    """
    
    api_key: str = ""
    api_base: str = "https://api.deepseek.com"
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    stop: Optional[List[str]] = None
    
    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """验证API密钥是否存在"""
        values["api_key"] = values.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
        if not values["api_key"]:
            raise ValueError(
                "DeepSeek API密钥未提供，请设置DEEPSEEK_API_KEY环境变量或直接传递api_key参数"
            )
        return values
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "deepseek"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """调用DeepSeek API"""
        
        # 合并stop序列
        if self.stop and stop:
            stop = list(set(self.stop + stop))
        elif self.stop:
            stop = self.stop
            
        # 请求体
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p
        }
        
        if stop:
            data["stop"] = stop
            
        # 添加其他参数
        for key, value in kwargs.items():
            data[key] = value
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                logger.error(f"DeepSeek API返回了无效的响应: {response_data}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"调用DeepSeek API时出错: {str(e)}")
            if response:
                logger.error(f"错误响应: {response.text}")
            raise ValueError(f"DeepSeek API调用失败: {str(e)}")
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> LLMResult:
        """生成多个回答"""
        generations = []
        
        for prompt in prompts:
            response = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([{"text": response}])
            
        return LLMResult(generations=generations)

class DeepSeekChat:
    """DeepSeek聊天模型的独立接口，支持多轮对话"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.deepseek.com",
        model_name: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None
    ):
        """
        初始化DeepSeek聊天模型
        
        Args:
            api_key: DeepSeek API密钥
            api_base: API基础URL
            model_name: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: Top-p采样参数
            system_prompt: 系统提示词
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API密钥未提供，请设置DEEPSEEK_API_KEY环境变量或直接传递api_key参数"
            )
            
        self.api_base = api_base
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # 初始化对话历史
        self.messages = []
        
        # 添加系统提示(如果有)
        if system_prompt:
            self.messages.append({
                "role": "system",
                "content": system_prompt
            })
    
    def add_message(self, role: str, content: str) -> None:
        """
        添加消息到对话历史
        
        Args:
            role: 消息角色 (user/assistant/system)
            content: 消息内容
        """
        self.messages.append({
            "role": role,
            "content": content
        })
    
    def chat(
        self, 
        message: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        发送消息并获取回复
        
        Args:
            message: 用户消息
            temperature: 温度参数
            max_tokens: 最大生成token数
            top_p: Top-p采样参数
            stop: 停止序列
            
        Returns:
            模型回复
        """
        # 添加用户消息
        self.add_message("user", message)
        
        # 请求体
        data = {
            "model": self.model_name,
            "messages": self.messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "top_p": top_p or self.top_p
        }
        
        if stop:
            data["stop"] = stop
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                assistant_message = response_data["choices"][0]["message"]["content"]
                
                # 添加助手消息到历史
                self.add_message("assistant", assistant_message)
                
                return assistant_message
            else:
                logger.error(f"DeepSeek API返回了无效的响应: {response_data}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"调用DeepSeek API时出错: {str(e)}")
            if 'response' in locals() and response:
                logger.error(f"错误响应: {response.text}")
            raise ValueError(f"DeepSeek API调用失败: {str(e)}")
    
    def clear_history(self, preserve_system_prompt: bool = True) -> None:
        """
        清除对话历史
        
        Args:
            preserve_system_prompt: 是否保留系统提示
        """
        if preserve_system_prompt and self.messages and self.messages[0]["role"] == "system":
            system_prompt = self.messages[0]
            self.messages = [system_prompt]
        else:
            self.messages = []
            
def get_deepseek_llm(
    api_key: Optional[str] = None,
    model_name: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.95
) -> DeepSeekLLM:
    """
    获取DeepSeek LLM实例
    
    Args:
        api_key: DeepSeek API密钥
        model_name: 模型名称
        temperature: 温度参数
        max_tokens: 最大生成token数
        top_p: Top-p采样参数
        
    Returns:
        DeepSeekLLM实例
    """
    return DeepSeekLLM(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    ) 
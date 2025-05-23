"""
DeepSeek LLM实现，基于DeepSeek API
"""
import os
import logging
import json
from typing import Any, Dict, List, Optional, Mapping, Union

import requests
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ChatMessage,
)

from config.config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_API_BASE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepSeekLLM(BaseChatModel):
    """DeepSeek LLM聊天模型"""
    
    api_key: str = Field(default=None, description="DeepSeek API密钥")
    model_name: str = Field(default=DEEPSEEK_MODEL, description="DeepSeek模型名称")
    api_base: str = Field(default=DEEPSEEK_API_BASE, description="DeepSeek API基础URL")
    temperature: float = Field(default=0.7, description="温度参数")
    max_tokens: int = Field(default=4096, description="最大生成token数")
    top_p: float = Field(default=0.95, description="top-p参数")
    
    class Config:
        """配置选项"""
        extra = Extra.forbid
    
    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """验证API密钥环境变量"""
        api_key = values.get("api_key") or os.getenv("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError(
                "DeepSeek API密钥未提供，请通过参数或环境变量DEEPSEEK_API_KEY设置"
            )
        values["api_key"] = api_key
        
        # 检查API基础URL
        api_base = values.get("api_base") or DEEPSEEK_API_BASE
        if not api_base:
            api_base = "https://api.deepseek.com/v1"
        
        # 确保API基础URL以/v1结尾
        if not api_base.endswith("/v1"):
            if api_base.endswith("/"):
                api_base = api_base + "v1"
            else:
                api_base = api_base + "/v1"
        
        values["api_base"] = api_base
        return values
    
    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "deepseek"
    
    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """将消息列表转换为DeepSeek API可用的格式"""
        prompt = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                prompt.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                prompt.append({"role": "assistant", "content": message.content})
            elif isinstance(message, ChatMessage):
                role = message.role
                # 映射角色
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                prompt.append({"role": role, "content": message.content})
            else:
                raise ValueError(f"不支持的消息类型: {type(message)}")
        
        return prompt
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成聊天完成"""
        # 合并参数
        params = {
            "model": self.model_name,
            "messages": self._convert_messages_to_prompt(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "top_p": kwargs.get("top_p", self.top_p),
        }
        
        # 添加停止词
        if stop:
            params["stop"] = stop
        
        # 记录请求信息
        logger.info(f"DeepSeek API请求: {self.api_base}/chat/completions")
        
        try:
            # 调用API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=params,
                timeout=120
            )
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"DeepSeek API错误: 状态码 {response.status_code}, 响应: {response.text}"
                logger.error(error_msg)
                if run_manager:
                    run_manager.on_llm_error(error=error_msg)
                raise ValueError(error_msg)
            
            # 解析响应
            response_data = response.json()
            
            # 提取生成的文本
            message = response_data.get("choices", [{}])[0].get("message", {})
            text = message.get("content", "")
            
            # 创建生成结果
            generation = ChatGeneration(
                message=AIMessage(content=text)
            )
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            error_msg = f"调用DeepSeek API时出错: {str(e)}"
            logger.error(error_msg)
            if run_manager:
                run_manager.on_llm_error(error=error_msg)
            raise
    
    def get_num_tokens(self, text: str) -> int:
        """估算文本的token数量"""
        # 简单估计：中文一个字约1.5个token，英文一个单词约1.3个token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        spaces = text.count(' ')
        other_chars = len(text) - chinese_chars - spaces
        
        # 估算token数
        estimated_tokens = chinese_chars * 1.5 + spaces + other_chars
        return int(estimated_tokens)
    
    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """估算消息列表的token数量"""
        prompt = self._convert_messages_to_prompt(messages)
        text = json.dumps(prompt)
        return self.get_num_tokens(text) 
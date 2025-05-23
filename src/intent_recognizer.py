#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
意图识别模块
基于LLM API实现用户查询意图分类
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
import json

# 导入LLM生成器
from generator import get_generator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntentRecognizer:
    """意图识别器类，用于识别用户查询的意图"""
    
    # 预定义意图类别
    INTENT_CLASSES = [
        "信息查询",    # 查询事实信息
        "比较类问题",  # 比较多个实体的异同
        "深度解释",    # 请求详细解释概念
        "推理分析",    # 需要逻辑推理的问题
        "操作指南",    # 如何操作/执行某事
        "个人观点",    # 请求观点或建议
        "闲聊",        # 非信息检索类闲聊
        "其他"         # 其他类型
    ]
    
    def __init__(self, llm_provider: str = "deepseek"):
        """
        初始化意图识别器
        
        Args:
            llm_provider: LLM提供商，支持openai和deepseek
        """
        self.llm_provider = llm_provider
        
        try:
            # 初始化LLM生成器
            self.generator = get_generator(provider=llm_provider)
            logger.info(f"意图识别器初始化完成，使用 {llm_provider} 提供商")
        except Exception as e:
            logger.error(f"初始化LLM生成器失败: {str(e)}")
            self.generator = None
            raise
    
    def recognize_intent(self, query: str) -> Dict[str, Any]:
        """
        识别用户查询的意图
        
        Args:
            query: 用户查询文本
            
        Returns:
            包含意图识别结果的字典
        """
        if self.generator is None:
            raise RuntimeError("LLM生成器未初始化")
        
        # 构建意图识别提示词
        prompt = f"""
请分析以下用户查询的意图类型，从给定的类别中选择最合适的一个。

用户查询: "{query}"

意图类别:
1. 信息查询 - 查询事实信息，如"什么是机器学习？"
2. 比较类问题 - 比较多个实体的异同，如"深度学习和机器学习的区别"
3. 深度解释 - 请求详细解释概念，如"详细解释神经网络的工作原理"
4. 推理分析 - 需要逻辑推理的问题，如"为什么深度学习在图像识别中效果更好？"
5. 操作指南 - 如何操作/执行某事，如"如何训练一个神经网络模型？"
6. 个人观点 - 请求观点或建议，如"你认为哪种算法更适合这个场景？"
7. 闲聊 - 非信息检索类闲聊，如"你好"、"谢谢"
8. 其他 - 其他类型

请回复JSON格式:
{{
    "intent": "意图类别名称",
    "confidence": 0.9,
    "reasoning": "选择这个意图的原因"
}}
"""
        
        try:
            # 调用LLM进行意图识别
            response = self.generator.llm(prompt)
            
            # 解析JSON响应
            try:
                # 尝试从响应中提取JSON部分
                if '{' in response and '}' in response:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    # 如果没有JSON格式，使用默认解析
                    result = self._parse_text_response(response, query)
                
                # 验证和标准化结果
                intent = result.get("intent", "其他")
                if intent not in self.INTENT_CLASSES:
                    intent = "其他"
                
                confidence = float(result.get("confidence", 0.8))
                if not 0 <= confidence <= 1:
                    confidence = 0.8
                
                reasoning = result.get("reasoning", "基于文本特征判断")
                
                final_result = {
                    "intent": intent,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "query": query
                }
                
                logger.info(f"意图识别成功: '{query}' -> '{intent}' (置信度: {confidence:.3f})")
                return final_result
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败，使用文本解析: {str(e)}")
                return self._parse_text_response(response, query)
                
        except Exception as e:
            logger.error(f"意图识别失败: {str(e)}")
            # 返回默认结果
            return {
                "intent": "其他",
                "confidence": 0.5,
                "reasoning": f"识别过程出错: {str(e)}",
                "query": query
            }
    
    def _parse_text_response(self, response: str, query: str) -> Dict[str, Any]:
        """
        解析文本响应，当JSON解析失败时使用
        
        Args:
            response: LLM的文本响应
            query: 原始查询
            
        Returns:
            意图识别结果
        """
        # 简单的关键词匹配逻辑
        response_lower = response.lower()
        query_lower = query.lower()
        
        # 基于关键词判断意图
        if any(keyword in query_lower for keyword in ["什么是", "是什么", "定义", "含义"]):
            intent = "信息查询"
            confidence = 0.8
        elif any(keyword in query_lower for keyword in ["区别", "不同", "差异", "比较", "对比"]):
            intent = "比较类问题"
            confidence = 0.8
        elif any(keyword in query_lower for keyword in ["为什么", "原因", "解释", "详细", "深入"]):
            intent = "深度解释"
            confidence = 0.7
        elif any(keyword in query_lower for keyword in ["分析", "推断", "判断", "预测", "评估"]):
            intent = "推理分析"
            confidence = 0.7
        elif any(keyword in query_lower for keyword in ["如何", "怎么", "步骤", "方法", "操作"]):
            intent = "操作指南"
            confidence = 0.8
        elif any(keyword in query_lower for keyword in ["建议", "推荐", "认为", "看法", "意见"]):
            intent = "个人观点"
            confidence = 0.7
        elif any(keyword in query_lower for keyword in ["你好", "谢谢", "再见", "聊天"]):
            intent = "闲聊"
            confidence = 0.9
        else:
            intent = "其他"
            confidence = 0.6
        
        return {
            "intent": intent,
            "confidence": confidence,
            "reasoning": f"基于关键词匹配判断: {intent}",
            "query": query
        }
    
    def get_retrieval_strategy(self, intent: str) -> Dict[str, Any]:
        """
        根据意图获取检索策略参数
        
        Args:
            intent: 识别的意图类型
            
        Returns:
            检索策略参数字典
        """
        # 根据不同意图设置不同的检索参数
        strategies = {
            "信息查询": {
                "top_k": 5,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "use_reranker": True,
                "temperature": 0.7
            },
            "比较类问题": {
                "top_k": 8,  # 比较类问题需要更多文档
                "vector_weight": 0.6,
                "bm25_weight": 0.4,
                "use_reranker": True,
                "temperature": 0.6
            },
            "深度解释": {
                "top_k": 6,
                "vector_weight": 0.8,  # 更注重语义
                "bm25_weight": 0.2,
                "use_reranker": True,
                "temperature": 0.5
            },
            "推理分析": {
                "top_k": 7,
                "vector_weight": 0.75,
                "bm25_weight": 0.25,
                "use_reranker": True,
                "temperature": 0.6
            },
            "操作指南": {
                "top_k": 4,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "use_reranker": True,
                "temperature": 0.4
            },
            "个人观点": {
                "top_k": 6,
                "vector_weight": 0.8,
                "bm25_weight": 0.2,
                "use_reranker": False,  # 个人观点可能不需要重排序
                "temperature": 0.8
            },
            "闲聊": {
                "top_k": 2,  # 闲聊不需要太多文档
                "vector_weight": 0.9,
                "bm25_weight": 0.1,
                "use_reranker": False,
                "temperature": 0.9
            },
            "其他": {
                "top_k": 5,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "use_reranker": True,
                "temperature": 0.7
            }
        }
        
        return strategies.get(intent, strategies["其他"])
    
    def get_prompt_template(self, intent: str) -> str:
        """
        根据意图获取对应的提示词模板
        
        Args:
            intent: 识别的意图类型
            
        Returns:
            提示词模板字符串
        """
        templates = {
            "信息查询": """基于以下相关文档内容，准确回答用户的问题。请提供具体、事实性的信息。

相关文档：
{context}

用户问题：{question}

请回答：""",
            
            "比较类问题": """基于以下相关文档内容，比较分析用户询问的内容。请从多个角度进行对比，突出关键差异和相似点。

相关文档：
{context}

用户问题：{question}

请进行详细比较分析：""",
            
            "深度解释": """基于以下相关文档内容，深入详细地解释用户询问的概念或现象。请提供全面的解释，包括原理、机制、应用等方面。

相关文档：
{context}

用户问题：{question}

请详细解释：""",
            
            "推理分析": """基于以下相关文档内容，运用逻辑推理分析用户的问题。请提供推理过程和结论。

相关文档：
{context}

用户问题：{question}

请进行推理分析：""",
            
            "操作指南": """基于以下相关文档内容，提供具体的操作指导。请给出清晰的步骤和注意事项。

相关文档：
{context}

用户问题：{question}

请提供操作指南：""",
            
            "个人观点": """基于以下相关文档内容，结合专业知识提供建议或观点。请给出有见地的分析和建议。

相关文档：
{context}

用户问题：{question}

请提供建议和观点：""",
            
            "闲聊": """请以友好、自然的方式回应用户。

用户说：{question}

请回应：""",
            
            "其他": """基于以下相关文档内容，回答用户的问题。

相关文档：
{context}

用户问题：{question}

请回答："""
        }
        
        return templates.get(intent, templates["其他"])

def get_intent_recognizer(llm_provider: str = "deepseek") -> IntentRecognizer:
    """
    获取意图识别器实例
    
    Args:
        llm_provider: LLM提供商
        
    Returns:
        IntentRecognizer实例
    """
    return IntentRecognizer(llm_provider=llm_provider) 
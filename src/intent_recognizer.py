#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
意图识别模块
基于OpenPrompt框架实现用户查询意图分类
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
import torch
from pathlib import Path

# 导入OpenPrompt相关模块
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer

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
    
    def __init__(self, model_name: str = "bert-base-chinese", device: str = None):
        """
        初始化意图识别器
        
        Args:
            model_name: 使用的预训练模型名称
            device: 运行设备，默认为自动选择
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练模型
        logger.info(f"加载预训练模型: {model_name}")
        self.plm, self.tokenizer, self.model_config, self.WrapperClass = load_plm("bert", model_name)
        
        # 定义模板
        self.template = ManualTemplate(
            text="用户问题: \"{'placeholder':'text_a'}\" \n这个问题的意图类型是什么? {'mask'}",
            tokenizer=self.tokenizer,
        )
        
        # 定义映射器
        self.verbalizer = ManualVerbalizer(
            classes=self.INTENT_CLASSES,
            label_words={
                "信息查询": ["查询", "了解", "是什么", "告诉我"],
                "比较类问题": ["比较", "区别", "不同", "差异", "优劣"],
                "深度解释": ["解释", "详细说明", "阐述", "为什么"],
                "推理分析": ["分析", "推理", "推断", "判断", "预测"],
                "操作指南": ["如何", "怎么样", "操作", "使用", "步骤"],
                "个人观点": ["认为", "建议", "意见", "看法", "推荐"],
                "闲聊": ["你好", "聊天", "问候", "闲聊", "感谢"],
                "其他": ["其他", "未知", "不确定"]
            },
            tokenizer=self.tokenizer,
        )
        
        # 构建意图识别模型
        self.model = PromptForClassification(
            template=self.template,
            plm=self.plm,
            verbalizer=self.verbalizer,
        )
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("意图识别器初始化完成")
    
    def recognize_intent(self, query: str) -> Dict[str, Any]:
        """
        识别用户查询的意图
        
        Args:
            query: 用户查询文本
            
        Returns:
            包含意图识别结果的字典
        """
        # 创建输入示例
        input_example = InputExample(text_a=query, guid=0)
        
        # 创建数据加载器
        data_loader = PromptDataLoader(
            dataset=[input_example],
            tokenizer=self.tokenizer,
            template=self.template,
            tokenizer_wrapper_class=self.WrapperClass,
            max_seq_length=512,
            batch_size=1,
        )
        
        # 进行预测
        with torch.no_grad():
            for batch in data_loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                logits = self.model(batch)
                preds = torch.softmax(logits, dim=-1)
                confidence, pred_id = torch.max(preds, dim=-1)
                intent = self.INTENT_CLASSES[pred_id.item()]
                confidence = confidence.item()
        
        # 返回意图和置信度
        result = {
            "intent": intent,
            "confidence": confidence,
            "all_intents": {
                self.INTENT_CLASSES[i]: preds[0][i].item()
                for i in range(len(self.INTENT_CLASSES))
            }
        }
        
        logger.info(f"查询: '{query}' -> 意图: '{intent}' (置信度: {confidence:.4f})")
        return result
    
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
                "use_reranker": True
            },
            "比较类问题": {
                "top_k": 8,  # 比较类问题需要更多文档
                "vector_weight": 0.6,
                "bm25_weight": 0.4,
                "use_reranker": True
            },
            "深度解释": {
                "top_k": 6,
                "vector_weight": 0.8,  # 更注重语义
                "bm25_weight": 0.2,
                "use_reranker": True
            },
            "推理分析": {
                "top_k": 7,
                "vector_weight": 0.75,
                "bm25_weight": 0.25,
                "use_reranker": True
            },
            "操作指南": {
                "top_k": 4,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "use_reranker": True
            },
            "个人观点": {
                "top_k": 6,
                "vector_weight": 0.8,
                "bm25_weight": 0.2,
                "use_reranker": False  # 个人观点可能不需要重排序
            },
            "闲聊": {
                "top_k": 2,  # 闲聊不需要太多文档
                "vector_weight": 0.9,
                "bm25_weight": 0.1,
                "use_reranker": False
            },
            "其他": {
                "top_k": 5,
                "vector_weight": 0.7,
                "bm25_weight": 0.3,
                "use_reranker": True
            }
        }
        
        return strategies.get(intent, strategies["其他"])
    
    def get_prompt_template(self, intent: str) -> str:
        """
        根据意图获取定制的提示模板
        
        Args:
            intent: 识别的意图类型
            
        Returns:
            适合该意图的提示模板
        """
        # 针对不同意图的提示模板
        templates = {
            "信息查询": """请基于以下参考资料回答用户的信息查询。只使用参考资料中的信息，如果参考资料不包含相关信息，请说明无法从参考资料中找到答案。
参考资料:
{context}

用户查询: {query}
回答:""",

            "比较类问题": """请根据以下参考资料，详细比较用户提问中涉及的对象或概念。对于每个比较点，请分别说明异同点，并使用表格或清晰的结构呈现。
参考资料:
{context}

用户的比较问题: {query}
比较分析:""",

            "深度解释": """请根据以下参考资料，对用户询问的概念或现象进行深入详细的解释。解释应包括核心定义、工作原理、相关背景和重要细节。
参考资料:
{context}

用户请求解释: {query}
详细解释:""",

            "推理分析": """请根据以下参考资料，对用户的问题进行逻辑推理和分析。需要综合考虑多个因素，并给出合理的推断过程和结论。
参考资料:
{context}

需要推理的问题: {query}
推理分析:""",

            "操作指南": """请根据以下参考资料，提供清晰、具体、分步骤的操作指南，回答用户关于如何执行特定任务的问题。
参考资料:
{context}

用户操作问题: {query}
操作指南:""",

            "个人观点": """请根据以下参考资料，提供一个平衡、有见地的观点或建议。注意既要基于事实，也要考虑不同角度的看法。
参考资料:
{context}

用户征求意见: {query}
建议与观点:""",

            "闲聊": """用户似乎在进行闲聊。请给予友好、有趣的回应，可以参考以下信息，但不必严格受限。
可参考信息:
{context}

用户闲聊: {query}
回应:"""
        }
        
        return templates.get(intent, templates["信息查询"])


def get_intent_recognizer(model_name: str = "bert-base-chinese") -> IntentRecognizer:
    """
    获取意图识别器实例
    
    Args:
        model_name: 使用的预训练模型名称
        
    Returns:
        意图识别器实例
    """
    return IntentRecognizer(model_name=model_name) 
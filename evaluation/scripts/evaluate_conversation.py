#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多轮对话评估脚本
专门用于评估RAG系统的指代补全和文档引用功能
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 添加项目根目录到Python路径
import sys
from pathlib import Path

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
# 获取项目根目录（evaluation/scripts的上两级目录）
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# 将项目根目录添加到Python路径
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import get_retriever
from src.generator import get_generator
from src.vectorizer import get_vectorizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定义目录常量
EVAL_DIR = SCRIPT_DIR.parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"

class ConversationEvaluator:
    """多轮对话评估器"""
    
    def __init__(self, generator, retriever=None):
        """
        初始化多轮对话评估器
        
        Args:
            generator: 生成器对象
            retriever: 检索器对象
        """
        self.generator = generator
        self.retriever = retriever
        logger.info("多轮对话评估器初始化完成")
    
    def evaluate_conversations(
        self, 
        conversations: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        评估多轮对话
        
        Args:
            conversations: 对话数据列表
            output_path: 评估结果保存路径
            
        Returns:
            评估结果
        """
        if not conversations:
            logger.error("对话数据为空")
            return {}
        
        results = []
        
        # 遍历对话
        for conv_data in conversations:
            conv_id = conv_data.get("id", "unknown")
            title = conv_data.get("title", "")
            conversation = conv_data.get("conversation", [])
            reference_answer = conv_data.get("reference_answer", "")
            expected_keywords = conv_data.get("expected_keywords", [])
            resolved_question = conv_data.get("resolved_question", "")
            
            logger.info(f"评估对话: {conv_id} - {title}")
            
            # 准备评估结果
            result = {
                "conversation_id": conv_id,
                "title": title,
                "original_conversation": conversation,
                "reference_answer": reference_answer,
                "expected_question": resolved_question,
            }
            
            # 重建对话历史
            chat_history = []
            for i in range(0, len(conversation) - 1, 2):
                if i + 1 < len(conversation):
                    chat_history.append(
                        (conversation[i]["content"], conversation[i+1]["content"])
                    )
            
            # 获取最后一个问题（可能包含指代词）
            last_question = conversation[-1]["content"] if conversation else ""
            
            # 生成回答
            generation_result = self.generator.generate_answer(
                query=last_question,
                chat_history=chat_history
            )
            
            # 提取结果
            answer = generation_result.get("answer", "")
            resolved_query = generation_result.get("resolved_query")
            original_query = generation_result.get("original_query", last_question)
            
            # 评估指代补全
            reference_resolution_score = self.evaluate_reference_resolution(
                resolved_query or original_query,
                resolved_question
            )
            
            # 评估回答质量
            answer_quality = self.evaluate_answer_quality(
                answer, 
                reference_answer,
                expected_keywords
            )
            
            # 评估引用质量
            citation_quality = self.evaluate_citation_quality(answer)
            
            # 组合评估结果
            result.update({
                "generated_answer": answer,
                "original_query": original_query,
                "resolved_query": resolved_query or original_query,
                "reference_resolution_score": reference_resolution_score,
                "answer_quality": answer_quality,
                "citation_quality": citation_quality,
                "overall_score": (
                    reference_resolution_score["score"] * 0.3 +
                    answer_quality["relevance"] * 0.4 +
                    citation_quality["score"] * 0.3
                )
            })
            
            results.append(result)
        
        # 计算整体评估指标
        avg_ref_resolution = sum(r["reference_resolution_score"]["score"] for r in results) / len(results)
        avg_answer_relevance = sum(r["answer_quality"]["relevance"] for r in results) / len(results)
        avg_citation_quality = sum(r["citation_quality"]["score"] for r in results) / len(results)
        avg_overall = sum(r["overall_score"] for r in results) / len(results)
        
        # 汇总结果
        evaluation_results = {
            "per_conversation_results": results,
            "average_metrics": {
                "reference_resolution": avg_ref_resolution,
                "answer_relevance": avg_answer_relevance,
                "citation_quality": avg_citation_quality,
                "overall_score": avg_overall
            },
            "total_conversations": len(results)
        }
        
        # 保存结果
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
                logger.info(f"评估结果已保存到: {output_path}")
            except Exception as e:
                logger.error(f"保存评估结果失败: {str(e)}")
        
        return evaluation_results
    
    def evaluate_reference_resolution(self, resolved_query: str, expected_query: str) -> Dict[str, Any]:
        """
        评估指代补全的质量
        
        Args:
            resolved_query: 系统补全后的问题
            expected_query: 预期的完整问题
            
        Returns:
            评估结果
        """
        # 简单基于文本相似度的评估
        if not resolved_query or not expected_query:
            return {"score": 0, "details": "查询为空"}
        
        # 分词并计算重叠度
        resolved_words = set(resolved_query.lower().replace("？", "").replace("?", "").split())
        expected_words = set(expected_query.lower().replace("？", "").replace("?", "").split())
        
        # 计算Jaccard相似度
        overlap = len(resolved_words.intersection(expected_words))
        union = len(resolved_words.union(expected_words))
        
        similarity = overlap / union if union > 0 else 0
        
        # 关键词覆盖
        key_words = set(w for w in expected_words if len(w) >= 2)
        covered_key_words = key_words.intersection(resolved_words)
        key_coverage = len(covered_key_words) / len(key_words) if key_words else 0
        
        # 最终得分
        score = 0.5 * similarity + 0.5 * key_coverage
        
        return {
            "score": score,
            "similarity": similarity,
            "key_coverage": key_coverage,
            "details": f"关键词覆盖: {len(covered_key_words)}/{len(key_words)}"
        }
    
    def evaluate_answer_quality(
        self, 
        answer: str, 
        reference: str, 
        expected_keywords: List[str]
    ) -> Dict[str, float]:
        """
        评估回答质量
        
        Args:
            answer: 生成的回答
            reference: 参考回答
            expected_keywords: 期望包含的关键词
            
        Returns:
            评估结果
        """
        if not answer or not reference:
            return {"relevance": 0, "keyword_coverage": 0}
        
        # 基于词语重叠的简单相关性评估
        answer_words = set(answer.lower().split())
        reference_words = set(reference.lower().split())
        
        overlap = len(answer_words.intersection(reference_words))
        relevance = overlap / len(reference_words) if reference_words else 0
        
        # 关键词覆盖
        keyword_coverage = 0
        if expected_keywords:
            covered_keywords = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
            keyword_coverage = covered_keywords / len(expected_keywords)
        
        return {
            "relevance": (relevance + keyword_coverage) / 2 if expected_keywords else relevance,
            "keyword_coverage": keyword_coverage
        }
    
    def evaluate_citation_quality(self, answer: str) -> Dict[str, Any]:
        """
        评估文档引用质量
        
        Args:
            answer: 生成的回答
            
        Returns:
            评估结果
        """
        if not answer:
            return {"score": 0, "has_citations": False, "citation_count": 0}
        
        # 检测是否包含引用标记，如[1]、[2]等
        import re
        citations = re.findall(r'\[\d+\]', answer)
        citation_count = len(citations)
        
        # 检测是否包含参考文献列表
        has_reference_list = "参考文献" in answer or "引用来源" in answer
        
        # 根据引用数量和是否有参考文献列表评分
        score = 0
        if citation_count > 0:
            score = 0.6  # 基础分
            score += min(0.3, citation_count * 0.1)  # 根据引用数量加分
            if has_reference_list:
                score += 0.1  # 有参考文献列表加分
        
        return {
            "score": score,
            "has_citations": citation_count > 0,
            "citation_count": citation_count,
            "has_reference_list": has_reference_list
        }

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多轮对话评估工具')
    parser.add_argument('--conversation_data', type=str, default=str(DATA_DIR / 'conversation_eval_dataset.json'),
                        help='对话数据集路径')
    parser.add_argument('--index_name', type=str, default='default',
                        help='向量存储索引名称')
    parser.add_argument('--output_dir', type=str, default=str(RESULTS_DIR),
                        help='评估结果输出目录')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 确保目录存在
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # 检查对话数据是否存在
    conv_data_path = Path(args.conversation_data)
    if not conv_data_path.exists():
        logger.error(f"对话数据集不存在: {conv_data_path}")
        logger.info("请先运行 python evaluation/scripts/create_conversation_eval_dataset.py 生成数据集")
        return
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载向量存储
    logger.info(f"初始化向量化器...")
    vectorizer = get_vectorizer()
    vector_store = vectorizer.load_vector_store(args.index_name)
    
    if vector_store is None:
        logger.error(f"无法加载向量存储: {args.index_name}")
        logger.info("请先构建向量库后再运行评估")
        return
    
    # 初始化检索器
    logger.info(f"初始化检索器...")
    retriever = get_retriever(vector_store=vector_store)
    
    # 初始化生成器
    logger.info(f"初始化生成器...")
    generator = get_generator(
        retriever=retriever,
        use_memory=True  # 确保启用对话记忆，以支持指代补全
    )
    
    # 加载对话数据
    logger.info(f"加载对话数据: {conv_data_path}")
    with open(conv_data_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # 初始化对话评估器
    evaluator = ConversationEvaluator(generator=generator, retriever=retriever)
    
    # 运行评估
    logger.info(f"开始评估，对话数量: {len(conversations)}")
    output_filename = output_dir / "conversation_evaluation_results.json"
    results = evaluator.evaluate_conversations(
        conversations=conversations,
        output_path=str(output_filename)
    )
    
    # 打印评估指标
    avg_metrics = results.get("average_metrics", {})
    print("\n===== 对话评估结果摘要 =====")
    print(f"总对话数: {results.get('total_conversations', 0)}")
    print(f"指代补全得分: {avg_metrics.get('reference_resolution', 0):.4f}")
    print(f"回答相关性得分: {avg_metrics.get('answer_relevance', 0):.4f}")
    print(f"引用质量得分: {avg_metrics.get('citation_quality', 0):.4f}")
    print(f"总体得分: {avg_metrics.get('overall_score', 0):.4f}")
    print(f"评估结果已保存至: {output_filename}")

if __name__ == "__main__":
    main() 
"""
评估模块，用于评估RAG系统的性能
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import StringEvaluator

from utils.helpers import time_function, ensure_directory
from config.config import BASE_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定义评估数据集路径
EVAL_DATA_DIR = BASE_DIR / "evaluation_data"
EVAL_RESULTS_DIR = BASE_DIR / "evaluation_results"

# 确保评估相关目录存在
EVAL_DATA_DIR.mkdir(exist_ok=True)
EVAL_RESULTS_DIR.mkdir(exist_ok=True)

class RAGEvaluator:
    """RAG系统评估器类，用于评估RAG系统的性能"""
    
    def __init__(
        self,
        retriever=None,
        generator=None,
        llm=None,
        eval_data_path: Optional[str] = None
    ):
        """
        初始化评估器
        
        Args:
            retriever: 检索器实例
            generator: 生成器实例
            llm: 语言模型实例
            eval_data_path: 评估数据集路径
        """
        self.retriever = retriever
        self.generator = generator
        self.llm = llm
        self.eval_data_path = eval_data_path
        self.eval_data = None
        
        # 加载评估数据
        if eval_data_path and os.path.exists(eval_data_path):
            self.load_eval_data(eval_data_path)
        
        # 初始化评估指标
        self.metrics = {
            "retrieval_precision": [],
            "retrieval_recall": [],
            "retrieval_f1": [],
            "answer_relevance": [],
            "answer_faithfulness": [],
            "answer_correctness": []
        }
        
        logger.info("评估器初始化完成")
        
    def load_eval_data(self, data_path: str) -> None:
        """
        加载评估数据集
        
        Args:
            data_path: 数据集路径
        """
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.eval_data = json.load(f)
            elif data_path.endswith('.csv'):
                self.eval_data = pd.read_csv(data_path).to_dict('records')
            elif data_path.endswith('.jsonl'):
                self.eval_data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.eval_data.append(json.loads(line))
            else:
                logger.error(f"不支持的数据文件格式: {data_path}")
                return
                
            logger.info(f"成功加载评估数据集: {data_path}，共 {len(self.eval_data)} 条数据")
        except Exception as e:
            logger.error(f"加载评估数据集失败: {str(e)}")
            self.eval_data = None
    
    def create_default_eval_data(self, output_path: Optional[str] = None) -> None:
        """
        创建默认评估数据集模板
        
        Args:
            output_path: 输出路径
        """
        # 生成示例评估数据
        default_eval_data = [
            {
                "question": "什么是RAG技术?",
                "reference_answer": "RAG(检索增强生成)是结合了检索系统和生成系统的技术，通过从知识库中检索相关信息来增强语言模型的生成能力，使回答更加准确可靠。",
                "reference_documents": ["doc1", "doc2"],
                "expected_keywords": ["检索", "生成", "知识库", "增强"]
            },
            {
                "question": "向量数据库有哪些选择?",
                "reference_answer": "常见的向量数据库包括FAISS、Chroma、Milvus、Pinecone和Weaviate等，它们各有特点和适用场景。",
                "reference_documents": ["doc3"],
                "expected_keywords": ["FAISS", "Chroma", "Milvus", "Pinecone", "Weaviate"]
            }
        ]
        
        # 确定输出路径
        if output_path is None:
            output_path = os.path.join(EVAL_DATA_DIR, "default_eval_data.json")
            
        # 保存到文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(default_eval_data, f, ensure_ascii=False, indent=2)
            logger.info(f"默认评估数据模板已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存默认评估数据模板失败: {str(e)}")
    
    @time_function
    def evaluate_retrieval(self, query: str, reference_docs: List[str]) -> Dict[str, float]:
        """
        评估检索性能
        
        Args:
            query: 查询文本
            reference_docs: 参考文档ID列表
            
        Returns:
            包含评估指标的字典
        """
        if self.retriever is None:
            logger.error("检索器未初始化，无法评估检索性能")
            return {"precision": 0, "recall": 0, "f1": 0}
            
        try:
            # 获取检索结果
            retrieved_docs = self.retriever.retrieve(query)
            retrieved_doc_ids = [doc.metadata.get("source", "") for doc in retrieved_docs]
            
            # 如果参考文档是完整路径，只保留文件名进行比较
            reference_doc_ids = [os.path.basename(doc) if os.path.isfile(doc) else doc for doc in reference_docs]
            retrieved_doc_ids = [os.path.basename(doc) if os.path.isfile(doc) else doc for doc in retrieved_doc_ids]
            
            # 计算精确率、召回率、F1
            true_positives = len(set(retrieved_doc_ids) & set(reference_doc_ids))
            retrieved_count = len(retrieved_doc_ids)
            relevant_count = len(reference_doc_ids)
            
            precision = true_positives / retrieved_count if retrieved_count > 0 else 0
            recall = true_positives / relevant_count if relevant_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {"precision": precision, "recall": recall, "f1": f1}
        except Exception as e:
            logger.error(f"评估检索性能时出错: {str(e)}")
            return {"precision": 0, "recall": 0, "f1": 0}
    
    @time_function
    def evaluate_answer(
        self, 
        query: str, 
        generated_answer: str, 
        reference_answer: str,
        expected_keywords: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        评估生成的回答质量
        
        Args:
            query: 查询文本
            generated_answer: 生成的回答
            reference_answer: 参考回答
            expected_keywords: 预期关键词列表
            
        Returns:
            包含评估指标的字典
        """
        try:
            results = {}
            
            # 计算关键词覆盖率
            if expected_keywords:
                covered_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in generated_answer.lower())
                keyword_coverage = covered_keywords / len(expected_keywords) if expected_keywords else 0
                results["keyword_coverage"] = keyword_coverage
            
            # 使用LangChain评估器进行评估（如果可用）
            if self.llm is not None:
                try:
                    # 相关性评估
                    relevance_evaluator = load_evaluator(
                        "relevance", llm=self.llm
                    )
                    relevance_score = relevance_evaluator.evaluate_strings(
                        prediction=generated_answer,
                        reference=query,
                    )
                    results["relevance"] = relevance_score.get("score", 0)
                    
                    # 忠实度评估
                    faithfulness_evaluator = load_evaluator(
                        "faithfulness", llm=self.llm
                    )
                    faithfulness_score = faithfulness_evaluator.evaluate_strings(
                        prediction=generated_answer,
                        reference=reference_answer,
                    )
                    results["faithfulness"] = faithfulness_score.get("score", 0)
                    
                    # 正确性评估
                    qa_evaluator = load_evaluator(
                        "qa", llm=self.llm
                    )
                    qa_score = qa_evaluator.evaluate_strings(
                        prediction=generated_answer,
                        reference=reference_answer,
                        input=query,
                    )
                    results["correctness"] = qa_score.get("score", 0)
                    
                except Exception as e:
                    logger.warning(f"使用LangChain评估器评估时出错: {str(e)}")
                    results.update({
                        "relevance": 0,
                        "faithfulness": 0,
                        "correctness": 0
                    })
            else:
                # 简单的基于重叠的评估方法
                # 计算文本相似度（简化版本）
                def simple_text_similarity(text1, text2):
                    words1 = set(text1.lower().split())
                    words2 = set(text2.lower().split())
                    overlap = len(words1.intersection(words2))
                    union = len(words1.union(words2))
                    return overlap / union if union > 0 else 0
                
                # 相关性：生成的回答与查询的相关性
                relevance = simple_text_similarity(query, generated_answer)
                # 忠实度：生成的回答与参考回答的相似度
                faithfulness = simple_text_similarity(generated_answer, reference_answer)
                # 正确性：基于相似度的简单估计
                correctness = faithfulness
                
                results.update({
                    "relevance": relevance,
                    "faithfulness": faithfulness,
                    "correctness": correctness
                })
            
            return results
        except Exception as e:
            logger.error(f"评估回答时出错: {str(e)}")
            return {
                "relevance": 0,
                "faithfulness": 0,
                "correctness": 0,
                "keyword_coverage": 0 if expected_keywords else None
            }
    
    @time_function
    def run_evaluation(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        运行完整的评估流程
        
        Args:
            output_path: 评估结果保存路径
            
        Returns:
            包含评估结果的字典
        """
        if self.eval_data is None:
            logger.error("评估数据未加载，无法进行评估")
            return {}
            
        if self.retriever is None or self.generator is None:
            logger.error("检索器或生成器未初始化，无法进行完整评估")
            return {}
            
        # 评估结果
        all_results = []
        
        # 遍历评估数据集
        for item in tqdm(self.eval_data, desc="评估进度"):
            question = item.get("question", "")
            reference_answer = item.get("reference_answer", "")
            reference_docs = item.get("reference_documents", [])
            expected_keywords = item.get("expected_keywords", [])
            
            # 单条评估结果
            result = {
                "question": question,
                "reference_answer": reference_answer
            }
            
            # 评估检索性能
            retrieval_metrics = self.evaluate_retrieval(question, reference_docs)
            result.update({
                "retrieval_precision": retrieval_metrics["precision"],
                "retrieval_recall": retrieval_metrics["recall"],
                "retrieval_f1": retrieval_metrics["f1"]
            })
            
            # 在全局指标中添加
            self.metrics["retrieval_precision"].append(retrieval_metrics["precision"])
            self.metrics["retrieval_recall"].append(retrieval_metrics["recall"])
            self.metrics["retrieval_f1"].append(retrieval_metrics["f1"])
            
            # 生成回答
            generation_result = self.generator.generate_answer(question)
            generated_answer = generation_result.get("answer", "")
            result["generated_answer"] = generated_answer
            
            # 评估回答质量
            answer_metrics = self.evaluate_answer(
                question, generated_answer, reference_answer, expected_keywords
            )
            
            result.update({
                "answer_relevance": answer_metrics.get("relevance", 0),
                "answer_faithfulness": answer_metrics.get("faithfulness", 0),
                "answer_correctness": answer_metrics.get("correctness", 0),
                "keyword_coverage": answer_metrics.get("keyword_coverage", 0) if expected_keywords else None
            })
            
            # 在全局指标中添加
            self.metrics["answer_relevance"].append(answer_metrics.get("relevance", 0))
            self.metrics["answer_faithfulness"].append(answer_metrics.get("faithfulness", 0))
            self.metrics["answer_correctness"].append(answer_metrics.get("correctness", 0))
            
            # 添加到总结果
            all_results.append(result)
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) if v else 0 for k, v in self.metrics.items()}
        
        # 整体评估结果
        evaluation_results = {
            "per_question_results": all_results,
            "average_metrics": avg_metrics,
            "total_questions": len(self.eval_data)
        }
        
        # 保存结果
        if output_path is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(EVAL_RESULTS_DIR, f"eval_results_{timestamp}.json")
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            logger.info(f"评估结果已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存评估结果失败: {str(e)}")
        
        return evaluation_results

def get_evaluator(
    retriever=None,
    generator=None,
    llm=None,
    eval_data_path: Optional[str] = None
) -> RAGEvaluator:
    """
    获取评估器实例
    
    Args:
        retriever: 检索器实例
        generator: 生成器实例
        llm: 语言模型实例
        eval_data_path: 评估数据集路径
        
    Returns:
        RAGEvaluator实例
    """
    return RAGEvaluator(
        retriever=retriever,
        generator=generator,
        llm=llm,
        eval_data_path=eval_data_path
    ) 
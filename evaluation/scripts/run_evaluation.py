#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG系统评估运行脚本
使用混合数据集评估RAG系统性能
"""
import os
import json
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
import sys
from pathlib import Path

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
# 获取项目根目录（evaluation/scripts的上两级目录）
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# 将项目根目录添加到Python路径
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import get_evaluator
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
EVAL_DATA_DIR = SCRIPT_DIR.parent / "data"
EVAL_RESULTS_DIR = SCRIPT_DIR.parent / "results"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RAG系统评估工具')
    parser.add_argument('--eval_data', type=str, default=str(EVAL_DATA_DIR / 'hybrid_eval_dataset.json'),
                        help='评估数据集路径')
    parser.add_argument('--index_name', type=str, default='default',
                        help='向量存储索引名称')
    parser.add_argument('--output_dir', type=str, default=str(EVAL_RESULTS_DIR),
                        help='评估结果输出目录')
    parser.add_argument('--top_k', type=int, default=5,
                        help='检索结果数量')
    parser.add_argument('--use_hybrid', action='store_true', default=True,
                        help='是否使用混合检索')
    parser.add_argument('--use_reranker', action='store_true', default=True,
                        help='是否使用重排序')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='评估样本数量，为None时使用全部数据')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 检查评估数据是否存在
    eval_data_path = Path(args.eval_data)
    if not eval_data_path.exists():
        logger.error(f"评估数据集不存在: {eval_data_path}")
        logger.info("请先运行 python evaluation/scripts/create_hybrid_eval_dataset.py 生成数据集")
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
    retriever = get_retriever(
        vector_store=vector_store,
        top_k=args.top_k,
        use_hybrid=args.use_hybrid,
        use_reranker=args.use_reranker
    )
    
    # 初始化生成器
    logger.info(f"初始化生成器...")
    generator = get_generator(retriever=retriever)
    
    # 加载评估数据
    logger.info(f"加载评估数据: {eval_data_path}")
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 样本抽样评估
    if args.sample_size is not None:
        import random
        if args.sample_size < len(eval_data):
            eval_data = random.sample(eval_data, args.sample_size)
            logger.info(f"从评估数据集中随机抽取{args.sample_size}个样本进行评估")
    
    # 初始化评估器
    logger.info(f"初始化评估器...")
    evaluator = get_evaluator(
        retriever=retriever,
        generator=generator,
        llm=generator.llm  # 使用生成器的语言模型
    )
    
    # 设置评估数据
    evaluator.eval_data = eval_data
    
    # 运行评估
    logger.info(f"开始评估，评估数据量: {len(eval_data)}")
    output_filename = output_dir / f"evaluation_results_{len(eval_data)}samples.json"
    results = evaluator.run_evaluation(output_path=str(output_filename))
    
    # 打印评估指标
    avg_metrics = results.get("average_metrics", {})
    print("\n===== 评估结果摘要 =====")
    print(f"总问题数: {results.get('total_questions', 0)}")
    print(f"检索精确率: {avg_metrics.get('retrieval_precision', 0):.4f}")
    print(f"检索召回率: {avg_metrics.get('retrieval_recall', 0):.4f}")
    print(f"检索F1分数: {avg_metrics.get('retrieval_f1', 0):.4f}")
    print(f"回答相关性: {avg_metrics.get('answer_relevance', 0):.4f}")
    print(f"回答忠实度: {avg_metrics.get('answer_faithfulness', 0):.4f}")
    print(f"回答正确性: {avg_metrics.get('answer_correctness', 0):.4f}")
    print(f"评估结果已保存至: {output_filename}")

if __name__ == "__main__":
    main() 
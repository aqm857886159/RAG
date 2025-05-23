#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合评估数据集生成脚本
结合CMRC（事实型问题）和C3（推理型问题）数据集
"""
import os
import json
import random
import logging
import requests
import pandas as pd
from tqdm import tqdm
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定义目录
EVAL_DIR = SCRIPT_DIR.parent
DATA_DIR = EVAL_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 数据集下载URL
CMRC_DEV_URL = "https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_dev.json"
C3_TRAIN_URL = "https://raw.githubusercontent.com/nlpdata/c3/master/data/c3-d-train.json"

def download_file(url, save_path):
    """下载文件到指定路径"""
    logger.info(f"下载数据集: {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 确保请求成功
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"文件已保存到: {save_path}")
        return True
    except Exception as e:
        logger.error(f"下载失败: {str(e)}")
        return False

def process_cmrc(input_file, output_list, max_samples=1000):
    """处理CMRC数据集"""
    logger.info(f"处理CMRC数据集: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            cmrc_data = json.load(f)
        
        count = 0
        for article in tqdm(cmrc_data['data'], desc="处理CMRC文章"):
            title = article['title']
            
            for para in article['paragraphs']:
                context = para['context']
                
                for qa in para['qas']:
                    question = qa['question']
                    answers = [a['text'] for a in qa['answers']]
                    reference_answer = answers[0] if answers else ""
                    
                    # 提取简单关键词
                    keywords = []
                    for word in reference_answer.replace("，", " ").replace("。", " ").split():
                        if len(word) >= 2 and word not in keywords:
                            keywords.append(word)
                    keywords = keywords[:5]  # 最多5个关键词
                    
                    # 创建评估条目
                    eval_item = {
                        "id": f"CMRC_{qa['id']}",
                        "category": "事实型",
                        "difficulty": "简单",
                        "question_type": "事实抽取",
                        "question": question,
                        "reference_answer": reference_answer,
                        "reference_documents": [title],
                        "context": context[:500] + "..." if len(context) > 500 else context,
                        "expected_keywords": keywords
                    }
                    
                    output_list.append(eval_item)
                    count += 1
                    
                    if count >= max_samples:
                        return count
        
        return count
    except Exception as e:
        logger.error(f"处理CMRC数据失败: {str(e)}")
        return 0

def process_c3(input_file, output_list, max_samples=500):
    """处理C3数据集"""
    logger.info(f"处理C3数据集: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            c3_data = json.load(f)
        
        count = 0
        for item in tqdm(c3_data, desc="处理C3数据"):
            context = "\n".join(item["context"])
            
            for qa in item["questions"]:
                question = qa["question"]
                options = qa["choice"]
                answer_idx = qa["answer"]
                answer = options[answer_idx]
                
                # 创建评估条目
                eval_item = {
                    "id": f"C3_{count}",
                    "category": "推理型",
                    "difficulty": "中等",
                    "question_type": "多项选择",
                    "question": question,
                    "reference_answer": answer,
                    "options": options,
                    "reference_documents": [f"C3_Doc_{count}"],
                    "context": context[:500] + "..." if len(context) > 500 else context,
                    "expected_keywords": [answer]
                }
                
                output_list.append(eval_item)
                count += 1
                
                if count >= max_samples:
                    return count
        
        return count
    except Exception as e:
        logger.error(f"处理C3数据失败: {str(e)}")
        return 0

def create_hybrid_dataset(cmrc_ratio=0.6, c3_ratio=0.4, total_samples=1000, output_file=None):
    """创建混合评估数据集"""
    if output_file is None:
        output_file = DATA_DIR / "hybrid_eval_dataset.json"
    
    # 计算每个数据集的样本数
    cmrc_samples = int(total_samples * cmrc_ratio)
    c3_samples = total_samples - cmrc_samples
    
    logger.info(f"创建混合数据集: CMRC {cmrc_samples}题 + C3 {c3_samples}题 = 共{total_samples}题")
    
    # 下载数据集
    cmrc_file = DATA_DIR / "cmrc2018_dev.json"
    c3_file = DATA_DIR / "c3_train.json"
    
    if not cmrc_file.exists():
        download_file(CMRC_DEV_URL, cmrc_file)
    
    if not c3_file.exists():
        download_file(C3_TRAIN_URL, c3_file)
    
    # 处理数据集
    eval_data = []
    
    cmrc_count = process_cmrc(cmrc_file, eval_data, max_samples=cmrc_samples)
    logger.info(f"已处理CMRC数据: {cmrc_count}题")
    
    c3_count = process_c3(c3_file, eval_data, max_samples=c3_samples)
    logger.info(f"已处理C3数据: {c3_count}题")
    
    # 打乱数据顺序
    random.shuffle(eval_data)
    
    # 保存为评估数据集格式
    rag_eval_data = []
    
    for item in eval_data:
        # 转换为RAG评估格式
        rag_item = {
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "reference_documents": item["reference_documents"],
            "expected_keywords": item["expected_keywords"],
            # 保留原始信息，方便后续分析
            "metadata": {
                "id": item["id"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "question_type": item["question_type"]
            }
        }
        
        # 对于C3类型的问题，添加选项信息
        if "options" in item:
            rag_item["metadata"]["options"] = item["options"]
        
        rag_eval_data.append(rag_item)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(rag_eval_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"混合评估数据集已生成: {output_file}")
    logger.info(f"共{len(rag_eval_data)}个问题 (CMRC: {cmrc_count}, C3: {c3_count})")
    
    # 生成数据集统计信息
    generate_stats(rag_eval_data, output_dir=DATA_DIR)
    
    return output_file

def generate_stats(eval_data, output_dir):
    """生成数据集统计信息"""
    categories = {}
    difficulties = {}
    question_types = {}
    
    for item in eval_data:
        metadata = item["metadata"]
        
        # 统计分类
        cat = metadata["category"]
        categories[cat] = categories.get(cat, 0) + 1
        
        # 统计难度
        diff = metadata["difficulty"]
        difficulties[diff] = difficulties.get(diff, 0) + 1
        
        # 统计问题类型
        q_type = metadata["question_type"]
        question_types[q_type] = question_types.get(q_type, 0) + 1
    
    # 创建统计报告
    report = {
        "total_questions": len(eval_data),
        "categories": categories,
        "difficulties": difficulties,
        "question_types": question_types
    }
    
    # 保存统计报告
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据集统计信息已生成: {stats_file}")
    
    # 打印简单统计
    print("\n数据集统计:")
    print(f"总问题数: {len(eval_data)}")
    print(f"问题类别: {categories}")
    print(f"难度分布: {difficulties}")
    print(f"问题类型: {question_types}")

if __name__ == "__main__":
    # 创建混合数据集，默认60%CMRC + 40%C3，总共1000题
    create_hybrid_dataset(cmrc_ratio=0.6, c3_ratio=0.4, total_samples=1000) 
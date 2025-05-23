#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG系统综合评估脚本
基于技术文档进行全面的RAG功能测试和性能评估
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 设置API密钥
os.environ["DEEPSEEK_API_KEY"] = "sk-06810fb5453e4fd1b39e3e5f566da210"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystemEvaluator:
    """RAG系统综合评估器"""
    
    def __init__(self):
        self.test_document = "Untitled.txt"
        self.evaluation_results = {}
        self.test_queries = [
            # 基础信息查询
            "什么是Streamlit？它有什么特点？",
            "这个项目的核心目标是什么？",
            "文档提到了哪些主要的技术栈？",
            
            # 技术细节查询
            "Unstructured.io有哪些关键特性？",
            "PDF解析推荐使用哪些库？为什么？", 
            "文本分块有哪些策略？各有什么优缺点？",
            
            # 对比分析查询
            "PyMuPDF和PDFPlumber的区别是什么？",
            "固定大小分块和语义分块的区别？",
            "OCR引擎中PaddleOCR和Tesseract有什么不同？",
            
            # 实施方案查询
            "项目实施分为哪几个阶段？每个阶段的重点是什么？",
            "部署有哪些方案选择？",
            "如何处理API密钥管理？",
            
            # 技术架构查询
            "系统的高层架构包含哪些层？",
            "UniversalDocumentProcessor采用了什么设计模式？",
            "异步处理有哪些方案？"
        ]
        
    def load_test_document(self) -> bool:
        """加载测试文档"""
        print("\n" + "="*70)
        print("📄 第一步：文档加载测试")
        print("="*70)
        
        try:
            from document_loader import DocumentLoader
            
            loader = DocumentLoader()
            
            if not Path(self.test_document).exists():
                print(f"❌ 测试文档不存在: {self.test_document}")
                return False
                
            start_time = time.time()
            docs = loader.load_single_document(self.test_document)
            load_time = time.time() - start_time
            
            print(f"✅ 文档加载成功")
            print(f"📊 文档片段数: {len(docs)}")
            print(f"⏱️ 加载耗时: {load_time:.2f} 秒")
            
            if docs:
                content_length = len(docs[0].page_content)
                print(f"📄 文档内容长度: {content_length:,} 字符")
                print(f"📝 内容预览: {docs[0].page_content[:200]}...")
                
            self.evaluation_results["document_loading"] = {
                "status": "success",
                "document_count": len(docs),
                "load_time": load_time,
                "content_length": content_length if docs else 0
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 文档加载失败: {str(e)}")
            self.evaluation_results["document_loading"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_text_chunking(self) -> bool:
        """测试文本分块功能"""
        print("\n" + "="*70)
        print("✂️ 第二步：文本分块测试")
        print("="*70)
        
        try:
            from document_loader import DocumentLoader
            from text_splitter import get_text_splitter
            
            loader = DocumentLoader()
            docs = loader.load_single_document(self.test_document)
            
            # 测试不同分块策略
            strategies = [
                ("recursive", {"chunk_size": 500, "chunk_overlap": 50}),
                ("recursive", {"chunk_size": 1000, "chunk_overlap": 100}),
                ("character", {"chunk_size": 500, "chunk_overlap": 50})
            ]
            
            chunking_results = {}
            
            for strategy_name, params in strategies:
                print(f"\n🔄 测试 {strategy_name} 分块策略 (chunk_size={params['chunk_size']})")
                
                start_time = time.time()
                splitter = get_text_splitter(strategy_name, **params)
                chunks = splitter.split_documents(docs)
                chunk_time = time.time() - start_time
                
                # 计算分块质量指标
                chunk_lengths = [len(chunk.page_content) for chunk in chunks]
                avg_length = statistics.mean(chunk_lengths)
                length_std = statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
                
                print(f"   ✅ 生成文本块: {len(chunks)} 个")
                print(f"   📊 平均长度: {avg_length:.0f} 字符")
                print(f"   📈 长度标准差: {length_std:.0f}")
                print(f"   ⏱️ 分块耗时: {chunk_time:.2f} 秒")
                
                # 显示示例块
                if chunks:
                    print(f"   📝 示例块: {chunks[0].page_content[:150]}...")
                
                chunking_results[f"{strategy_name}_{params['chunk_size']}"] = {
                    "chunk_count": len(chunks),
                    "avg_length": avg_length,
                    "length_std": length_std,
                    "chunk_time": chunk_time
                }
            
            self.evaluation_results["text_chunking"] = {
                "status": "success",
                "strategies_tested": len(strategies),
                "results": chunking_results
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 文本分块测试失败: {str(e)}")
            self.evaluation_results["text_chunking"] = {
                "status": "failed", 
                "error": str(e)
            }
            return False
    
    def test_intent_recognition(self) -> bool:
        """测试意图识别准确性"""
        print("\n" + "="*70) 
        print("🧠 第三步：意图识别测试")
        print("="*70)
        
        try:
            from intent_recognizer import get_intent_recognizer
            
            recognizer = get_intent_recognizer(llm_provider="deepseek")
            
            intent_results = []
            total_time = 0
            
            # 预期意图分类（用于评估准确性）
            expected_intents = {
                "什么是Streamlit？它有什么特点？": "信息查询",
                "这个项目的核心目标是什么？": "信息查询", 
                "PyMuPDF和PDFPlumber的区别是什么？": "比较类问题",
                "固定大小分块和语义分块的区别？": "比较类问题",
                "项目实施分为哪几个阶段？每个阶段的重点是什么？": "操作指南",
                "如何处理API密钥管理？": "操作指南"
            }
            
            correct_predictions = 0
            
            for query in expected_intents.keys():
                print(f"\n🔄 测试查询: {query[:50]}...")
                
                start_time = time.time()
                result = recognizer.recognize_intent(query)
                intent_time = time.time() - start_time
                total_time += intent_time
                
                predicted_intent = result['intent']
                expected_intent = expected_intents[query]
                is_correct = predicted_intent == expected_intent
                
                if is_correct:
                    correct_predictions += 1
                    print(f"   ✅ 预测正确: {predicted_intent} (置信度: {result['confidence']:.3f})")
                else:
                    print(f"   ⚠️ 预测错误: {predicted_intent} (期望: {expected_intent}, 置信度: {result['confidence']:.3f})")
                
                intent_results.append({
                    "query": query,
                    "predicted": predicted_intent,
                    "expected": expected_intent,
                    "correct": is_correct,
                    "confidence": result['confidence'],
                    "time": intent_time
                })
            
            accuracy = correct_predictions / len(expected_intents)
            avg_time = total_time / len(expected_intents)
            avg_confidence = statistics.mean([r['confidence'] for r in intent_results])
            
            print(f"\n📊 意图识别评估结果:")
            print(f"   🎯 准确率: {accuracy:.2%} ({correct_predictions}/{len(expected_intents)})")
            print(f"   ⏱️ 平均耗时: {avg_time:.2f} 秒/次")
            print(f"   📈 平均置信度: {avg_confidence:.3f}")
            
            self.evaluation_results["intent_recognition"] = {
                "status": "success",
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_queries": len(expected_intents),
                "avg_time": avg_time,
                "avg_confidence": avg_confidence,
                "detailed_results": intent_results
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 意图识别测试失败: {str(e)}")
            self.evaluation_results["intent_recognition"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_qa_generation(self) -> bool:
        """测试问答对生成质量"""
        print("\n" + "="*70)
        print("💬 第四步：问答对生成测试")
        print("="*70)
        
        try:
            from data_converter import DocumentConverter
            from document_loader import DocumentLoader
            
            # 加载文档并转换
            loader = DocumentLoader()
            docs = loader.load_single_document(self.test_document)
            
            converter = DocumentConverter(
                llm_provider="deepseek",
                api_key="sk-06810fb5453e4fd1b39e3e5f566da210"
            )
            
            # 分割文档（取前几个块进行测试以节省时间）
            chunks = converter.text_splitter.split_documents(docs)
            test_chunks = chunks[:3]  # 只测试前3个块
            
            print(f"🔄 对 {len(test_chunks)} 个文本块进行问答对生成测试...")
            
            start_time = time.time()
            qa_pairs = converter._generate_qa_pairs(test_chunks)
            generation_time = time.time() - start_time
            
            print(f"✅ 问答对生成完成")
            print(f"📊 生成数量: {len(qa_pairs)} 个问答对")
            print(f"⏱️ 生成耗时: {generation_time:.2f} 秒")
            print(f"📈 生成效率: {len(qa_pairs)/generation_time:.1f} 个/秒")
            
            # 显示生成的问答对示例
            print(f"\n📝 生成的问答对示例:")
            for i, qa in enumerate(qa_pairs[:3]):  # 显示前3个
                print(f"\n   Q{i+1}: {qa.question}")
                print(f"   A{i+1}: {qa.answer[:150]}{'...' if len(qa.answer) > 150 else ''}")
            
            # 评估问答对质量（基于简单指标）
            avg_question_length = statistics.mean([len(qa.question) for qa in qa_pairs])
            avg_answer_length = statistics.mean([len(qa.answer) for qa in qa_pairs])
            
            self.evaluation_results["qa_generation"] = {
                "status": "success",
                "qa_pairs_count": len(qa_pairs),
                "generation_time": generation_time,
                "generation_rate": len(qa_pairs)/generation_time,
                "avg_question_length": avg_question_length,
                "avg_answer_length": avg_answer_length,
                "sample_qa_pairs": [
                    {"question": qa.question, "answer": qa.answer} 
                    for qa in qa_pairs[:2]
                ]
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 问答对生成测试失败: {str(e)}")
            self.evaluation_results["qa_generation"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_vector_retrieval(self) -> bool:
        """测试向量检索相关性"""
        print("\n" + "="*70)
        print("🔍 第五步：向量检索测试")
        print("="*70)
        
        try:
            from vectorizer import SimpleMemoryVectorStore
            from sentence_transformers import SentenceTransformer
            from document_loader import DocumentLoader
            from text_splitter import get_text_splitter
            
            # 创建适配器
            class SentenceTransformerAdapter:
                def __init__(self, model):
                    self.model = model
                    
                def embed_documents(self, texts):
                    return self.model.encode(texts).tolist()
                    
                def embed_query(self, text):
                    return self.model.encode([text])[0].tolist()
            
            # 初始化组件
            print("🔄 初始化向量存储系统...")
            base_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = SentenceTransformerAdapter(base_model)
            vector_store = SimpleMemoryVectorStore(embeddings)
            
            # 加载和分块文档
            loader = DocumentLoader()
            docs = loader.load_single_document(self.test_document)
            splitter = get_text_splitter("recursive", chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            
            # 构建向量存储（使用前20个块避免过长）
            test_chunks = chunks[:20]
            print(f"🔄 构建向量存储 ({len(test_chunks)} 个文本块)...")
            
            build_start = time.time()
            vector_store.add_documents(test_chunks)
            build_time = time.time() - build_start
            
            print(f"✅ 向量存储构建完成，耗时 {build_time:.2f} 秒")
            
            # 测试检索相关性
            test_queries_for_retrieval = [
                "Streamlit的特点和优势",
                "文档解析的技术选型", 
                "文本分块策略比较",
                "部署方案选择"
            ]
            
            retrieval_results = []
            total_retrieval_time = 0
            
            for query in test_queries_for_retrieval:
                print(f"\n🔍 检索查询: {query}")
                
                start_time = time.time()
                results = vector_store.similarity_search(query, k=3)
                search_time = time.time() - start_time
                total_retrieval_time += search_time
                
                print(f"   ✅ 找到 {len(results)} 个相关文档，耗时 {search_time:.3f} 秒")
                
                # 显示最相关的结果
                if results:
                    print(f"   📄 最相关内容: {results[0].page_content[:100]}...")
                
                retrieval_results.append({
                    "query": query,
                    "results_count": len(results),
                    "search_time": search_time,
                    "top_result": results[0].page_content[:200] if results else ""
                })
            
            avg_retrieval_time = total_retrieval_time / len(test_queries_for_retrieval)
            
            print(f"\n📊 向量检索评估结果:")
            print(f"   📚 文档库大小: {len(test_chunks)} 个文档块")
            print(f"   🏗️ 构建耗时: {build_time:.2f} 秒")
            print(f"   🔍 平均检索耗时: {avg_retrieval_time:.3f} 秒/次")
            print(f"   ⚡ 检索速度: {len(test_chunks)/avg_retrieval_time:.0f} 文档/秒")
            
            self.evaluation_results["vector_retrieval"] = {
                "status": "success",
                "document_count": len(test_chunks),
                "build_time": build_time,
                "avg_retrieval_time": avg_retrieval_time,
                "queries_tested": len(test_queries_for_retrieval),
                "retrieval_results": retrieval_results
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 向量检索测试失败: {str(e)}")
            self.evaluation_results["vector_retrieval"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def test_end_to_end_rag(self) -> bool:
        """测试端到端RAG问答"""
        print("\n" + "="*70)
        print("🤖 第六步：端到端RAG问答测试")
        print("="*70)
        
        try:
            from generator import get_generator
            from src.llm import get_llm
            from vectorizer import SimpleMemoryVectorStore
            from sentence_transformers import SentenceTransformer
            from document_loader import DocumentLoader
            from text_splitter import get_text_splitter
            
            # 初始化完整RAG系统
            print("🔄 初始化完整RAG系统...")
            
            # 创建适配器
            class SentenceTransformerAdapter:
                def __init__(self, model):
                    self.model = model
                def embed_documents(self, texts):
                    return self.model.encode(texts).tolist()
                def embed_query(self, text):
                    return self.model.encode([text])[0].tolist()
            
            # 构建知识库
            loader = DocumentLoader()
            docs = loader.load_single_document(self.test_document)
            splitter = get_text_splitter("recursive", chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            
            # 向量存储
            base_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = SentenceTransformerAdapter(base_model)
            vector_store = SimpleMemoryVectorStore(embeddings)
            vector_store.add_documents(chunks[:15])  # 使用前15个块
            
            # 创建RAG生成器
            llm = get_llm(provider="deepseek", api_key="sk-06810fb5453e4fd1b39e3e5f566da210")
            rag_generator = get_generator(use_rag=True, llm=llm, vector_store=vector_store)
            
            print("✅ RAG系统初始化完成")
            
            # 测试查询
            test_rag_queries = [
                "Streamlit有什么特点？为什么选择它？",
                "PDF解析推荐使用哪个库？原因是什么？",
                "项目实施分为哪些阶段？"
            ]
            
            rag_results = []
            total_rag_time = 0
            
            for query in test_rag_queries:
                print(f"\n❓ 问题: {query}")
                
                start_time = time.time()
                result = rag_generator.generate(query)
                rag_time = time.time() - start_time
                total_rag_time += rag_time
                
                answer = result.get('answer', '未生成回答')
                source_docs = result.get('source_documents', [])
                
                print(f"   ✅ 回答生成完成，耗时 {rag_time:.2f} 秒")
                print(f"   📚 使用了 {len(source_docs)} 个相关文档")
                print(f"   💬 回答: {answer[:200]}{'...' if len(answer) > 200 else ''}")
                
                rag_results.append({
                    "query": query,
                    "answer": answer,
                    "source_count": len(source_docs),
                    "response_time": rag_time,
                    "answer_length": len(answer)
                })
            
            avg_rag_time = total_rag_time / len(test_rag_queries)
            avg_answer_length = statistics.mean([r['answer_length'] for r in rag_results])
            
            print(f"\n📊 端到端RAG评估结果:")
            print(f"   🎯 测试查询数: {len(test_rag_queries)}")
            print(f"   ⏱️ 平均响应时间: {avg_rag_time:.2f} 秒/次") 
            print(f"   📝 平均回答长度: {avg_answer_length:.0f} 字符")
            print(f"   📚 知识库大小: {len(chunks[:15])} 个文档块")
            
            self.evaluation_results["end_to_end_rag"] = {
                "status": "success",
                "queries_tested": len(test_rag_queries),
                "avg_response_time": avg_rag_time,
                "avg_answer_length": avg_answer_length,
                "knowledge_base_size": len(chunks[:15]),
                "detailed_results": rag_results
            }
            
            return True
            
        except Exception as e:
            print(f"❌ 端到端RAG测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.evaluation_results["end_to_end_rag"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合评估报告"""
        print("\n" + "="*70)
        print("📊 RAG系统综合评估报告")
        print("="*70)
        
        # 计算总体评分
        passed_tests = 0
        total_tests = 0
        
        for test_name, result in self.evaluation_results.items():
            total_tests += 1
            if result.get("status") == "success":
                passed_tests += 1
        
        overall_score = passed_tests / total_tests if total_tests > 0 else 0
        
        # 性能指标汇总
        performance_metrics = {}
        
        if "document_loading" in self.evaluation_results:
            doc_result = self.evaluation_results["document_loading"]
            if doc_result.get("status") == "success":
                performance_metrics["文档加载"] = {
                    "耗时": f"{doc_result.get('load_time', 0):.2f}秒",
                    "文档长度": f"{doc_result.get('content_length', 0):,}字符"
                }
        
        if "intent_recognition" in self.evaluation_results:
            intent_result = self.evaluation_results["intent_recognition"] 
            if intent_result.get("status") == "success":
                performance_metrics["意图识别"] = {
                    "准确率": f"{intent_result.get('accuracy', 0):.1%}",
                    "平均耗时": f"{intent_result.get('avg_time', 0):.2f}秒",
                    "平均置信度": f"{intent_result.get('avg_confidence', 0):.3f}"
                }
        
        if "qa_generation" in self.evaluation_results:
            qa_result = self.evaluation_results["qa_generation"]
            if qa_result.get("status") == "success":
                performance_metrics["问答生成"] = {
                    "生成数量": f"{qa_result.get('qa_pairs_count', 0)}个",
                    "生成效率": f"{qa_result.get('generation_rate', 0):.1f}个/秒"
                }
        
        if "vector_retrieval" in self.evaluation_results:
            retrieval_result = self.evaluation_results["vector_retrieval"]
            if retrieval_result.get("status") == "success":
                performance_metrics["向量检索"] = {
                    "检索耗时": f"{retrieval_result.get('avg_retrieval_time', 0):.3f}秒",
                    "文档库大小": f"{retrieval_result.get('document_count', 0)}个块"
                }
        
        if "end_to_end_rag" in self.evaluation_results:
            rag_result = self.evaluation_results["end_to_end_rag"]
            if rag_result.get("status") == "success":
                performance_metrics["端到端RAG"] = {
                    "响应时间": f"{rag_result.get('avg_response_time', 0):.2f}秒",
                    "回答长度": f"{rag_result.get('avg_answer_length', 0):.0f}字符"
                }
        
        # 打印报告
        print(f"\n🎯 总体评估:")
        print(f"   测试通过率: {overall_score:.1%} ({passed_tests}/{total_tests})")
        
        if overall_score >= 0.8:
            print(f"   系统状态: 🟢 优秀 - RAG系统运行状态良好")
        elif overall_score >= 0.6:
            print(f"   系统状态: 🟡 良好 - RAG系统基本可用，部分功能需优化")
        else:
            print(f"   系统状态: 🔴 需改进 - RAG系统存在较多问题")
        
        print(f"\n📈 性能指标汇总:")
        for category, metrics in performance_metrics.items():
            print(f"   {category}:")
            for metric, value in metrics.items():
                print(f"     - {metric}: {value}")
        
        # 生成建议
        recommendations = []
        
        if "intent_recognition" in self.evaluation_results:
            intent_result = self.evaluation_results["intent_recognition"]
            if intent_result.get("status") == "success":
                accuracy = intent_result.get("accuracy", 0)
                if accuracy < 0.8:
                    recommendations.append("建议优化意图识别提示词以提高准确率")
                if intent_result.get("avg_time", 0) > 5:
                    recommendations.append("意图识别响应时间较长，建议优化LLM调用")
        
        if "vector_retrieval" in self.evaluation_results:
            retrieval_result = self.evaluation_results["vector_retrieval"]
            if retrieval_result.get("status") == "success":
                if retrieval_result.get("avg_retrieval_time", 0) > 0.1:
                    recommendations.append("向量检索速度可进一步优化")
        
        if "end_to_end_rag" in self.evaluation_results:
            rag_result = self.evaluation_results["end_to_end_rag"]
            if rag_result.get("status") == "success":
                if rag_result.get("avg_response_time", 0) > 10:
                    recommendations.append("RAG响应时间较长，建议优化检索和生成流程")
        
        if not recommendations:
            recommendations.append("系统运行良好，建议持续监控性能指标")
        
        print(f"\n💡 优化建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # 返回完整报告
        report = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_document": self.test_document,
            "overall_score": overall_score,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "performance_metrics": performance_metrics,
            "recommendations": recommendations,
            "detailed_results": self.evaluation_results
        }
        
        return report
    
    def run_evaluation(self) -> Dict[str, Any]:
        """运行完整评估流程"""
        print("🚀 开始RAG系统综合评估")
        print(f"📅 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📄 测试文档: {self.test_document}")
        
        start_time = time.time()
        
        # 执行各项测试
        tests = [
            ("文档加载", self.load_test_document),
            ("文本分块", self.test_text_chunking),
            ("意图识别", self.test_intent_recognition),
            ("问答生成", self.test_qa_generation),
            ("向量检索", self.test_vector_retrieval),
            ("端到端RAG", self.test_end_to_end_rag),
        ]
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                if not success:
                    print(f"⚠️ {test_name}测试未完全通过")
            except Exception as e:
                print(f"❌ {test_name}测试执行异常: {str(e)}")
        
        total_time = time.time() - start_time
        print(f"\n⏱️ 总评估耗时: {total_time:.2f} 秒")
        
        # 生成综合报告
        report = self.generate_comprehensive_report()
        
        return report

def main():
    """主函数"""
    evaluator = RAGSystemEvaluator()
    report = evaluator.run_evaluation()
    
    # 保存评估报告
    report_file = f"rag_evaluation_report_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 评估报告已保存: {report_file}")
    return report

if __name__ == "__main__":
    main() 
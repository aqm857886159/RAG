#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修复后的功能
验证意图识别和问答对生成是否正常工作
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_intent_recognition():
    """测试意图识别功能"""
    print("\n" + "="*50)
    print("🧠 测试意图识别功能")
    print("="*50)
    
    try:
        from intent_recognizer import get_intent_recognizer
        
        # 初始化意图识别器
        recognizer = get_intent_recognizer(llm_provider="deepseek")
        
        # 测试查询列表
        test_queries = [
            "什么是机器学习？",
            "深度学习和机器学习的区别是什么？", 
            "详细解释神经网络的工作原理",
            "为什么深度学习在图像识别中效果更好？",
            "如何训练一个神经网络模型？",
            "你认为哪种算法更适合这个场景？",
            "你好，今天天气怎么样？"
        ]
        
        results = []
        for query in test_queries:
            print(f"\n📝 测试查询: {query}")
            try:
                result = recognizer.recognize_intent(query)
                print(f"   ✅ 意图: {result['intent']}")
                print(f"   📊 置信度: {result['confidence']:.3f}")
                print(f"   💭 推理: {result['reasoning']}")
                
                results.append(result)
                
                # 测试检索策略
                strategy = recognizer.get_retrieval_strategy(result['intent'])
                print(f"   🎯 检索策略: top_k={strategy['top_k']}, vector_weight={strategy['vector_weight']}")
                
            except Exception as e:
                print(f"   ❌ 错误: {str(e)}")
                
        print(f"\n✅ 意图识别测试完成，成功处理 {len(results)}/{len(test_queries)} 个查询")
        return len(results) == len(test_queries)
        
    except Exception as e:
        print(f"❌ 意图识别测试失败: {str(e)}")
        return False

def test_qa_generation():
    """测试问答对生成功能"""
    print("\n" + "="*50)
    print("💬 测试问答对生成功能")
    print("="*50)
    
    try:
        from data_converter import DocumentConverter
        
        # 初始化文档转换器
        converter = DocumentConverter(
            llm_provider="deepseek",
            api_key="sk-06810fb5453e4fd1b39e3e5f566da210"
        )
        
        # 查找测试文档 - 修正路径
        test_files = (list(Path("docs").glob("*.docx")) + 
                     list(Path("docs").glob("*.pdf")) + 
                     list(Path("docs").glob("*.txt")) +
                     list(Path("data").glob("*.docx")) + 
                     list(Path("data").glob("*.pdf")) + 
                     list(Path("data").glob("*.txt")))
        
        if not test_files:
            # 如果没有找到文档，创建一个临时测试文档
            print("📝 创建临时测试文档")
            from langchain.docstore.document import Document
            
            # 直接使用文本内容创建文档
            test_content = """机器学习基础知识

什么是机器学习？
机器学习是人工智能的一个分支，它使用算法和统计模型来使计算机系统能够从数据中学习和做出决策，而无需明确的编程指令。

深度学习和机器学习的区别
深度学习是机器学习的一个子集，它使用人工神经网络来模拟人脑的学习过程。主要区别包括：
1. 深度学习使用多层神经网络
2. 深度学习可以自动提取特征
3. 深度学习在处理大量数据时表现更好"""
            
            # 创建文档对象
            docs = [Document(page_content=test_content, metadata={"source": "test_text"})]
            
            # 分割文本
            chunks = converter.text_splitter.split_documents(docs)
            print(f"📄 使用临时测试内容，共 {len(chunks)} 个文本块")
            
            # 生成问答对
            qa_pairs = converter._generate_qa_pairs(chunks)
            
            print(f"📊 处理结果:")
            print(f"   - 文本块数量: {len(chunks)}")
            print(f"   - 问答对数量: {len(qa_pairs)}")
            
            # 显示前几个问答对
            if qa_pairs:
                print(f"\n💬 生成的问答对示例:")
                for i, qa in enumerate(qa_pairs[:3]):
                    print(f"\n{i+1}. 问题: {qa.question}")
                    print(f"   答案: {qa.answer}")
            else:
                print("❌ 未生成任何问答对")
                return False
                
            print(f"\n✅ 问答对生成测试完成，生成了 {len(qa_pairs)} 个问答对")
            return len(qa_pairs) > 0
        else:
            test_file = test_files[0]
            print(f"📄 使用测试文档: {test_file}")
            
            # 处理文档并生成问答对
            result = converter.process_document(str(test_file), output_formats=["json", "qa"])
            
            print(f"📊 处理结果:")
            print(f"   - 文本块数量: {len(result.text_chunks)}")
            print(f"   - 问答对数量: {len(result.qa_pairs)}")
            
            # 显示前几个问答对
            if result.qa_pairs:
                print(f"\n💬 生成的问答对示例:")
                for i, qa in enumerate(result.qa_pairs[:3]):
                    print(f"\n{i+1}. 问题: {qa.question}")
                    print(f"   答案: {qa.answer}")
            else:
                print("❌ 未生成任何问答对")
                return False
                
            print(f"\n✅ 问答对生成测试完成，生成了 {len(result.qa_pairs)} 个问答对")
            return len(result.qa_pairs) > 0
        
    except Exception as e:
        print(f"❌ 问答对生成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_with_intent():
    """测试集成意图识别的RAG系统"""
    print("\n" + "="*50)
    print("🔍 测试集成意图识别的RAG系统")
    print("="*50)
    
    try:
        # 尝试不同的模块名
        retrieval_system = None
        intent_recognizer_module = None
        
        try:
            from retriever import SimpleRetriever
            retrieval_system = SimpleRetriever()
        except ImportError:
            try:
                from vectorizer import VectorDatabase
                retrieval_system = VectorDatabase()
            except ImportError:
                print("❌ 未找到可用的检索系统模块")
                return False
        
        from intent_recognizer import get_intent_recognizer
        intent_recognizer = get_intent_recognizer(llm_provider="deepseek")
        
        # 测试查询
        test_query = "深度学习和机器学习的区别是什么？"
        print(f"🤔 测试查询: {test_query}")
        
        # 识别意图
        intent_result = intent_recognizer.recognize_intent(test_query)
        print(f"🧠 识别意图: {intent_result['intent']} (置信度: {intent_result['confidence']:.3f})")
        print(f"💭 推理: {intent_result['reasoning']}")
        
        # 获取检索策略
        strategy = intent_recognizer.get_retrieval_strategy(intent_result['intent'])
        print(f"🎯 检索策略: {strategy}")
        
        # 获取提示词模板
        template = intent_recognizer.get_prompt_template(intent_result['intent'])
        print(f"📝 提示词模板: {template[:100]}...")
        
        print(f"\n✅ 意图识别集成测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试修复后的功能")
    
    # 检查API密钥
    if not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "sk-06810fb5453e4fd1b39e3e5f566da210"
    
    results = []
    
    # 测试1: 意图识别
    print("\n" + "🔹"*20 + " 测试1: 意图识别 " + "🔹"*20)
    results.append(("意图识别", test_intent_recognition()))
    
    # 测试2: 问答对生成  
    print("\n" + "🔹"*20 + " 测试2: 问答对生成 " + "🔹"*20)
    results.append(("问答对生成", test_qa_generation()))
    
    # 测试3: 集成RAG系统
    print("\n" + "🔹"*20 + " 测试3: 集成RAG系统 " + "🔹"*20)
    results.append(("集成RAG", test_rag_with_intent()))
    
    # 总结测试结果
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有功能修复成功！")
    else:
        print("⚠️  部分功能仍需要修复")
        
    return passed == total

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心功能测试脚本
专门测试已修复的意图识别和问答对生成功能
"""

import os
import sys
import logging
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 设置API密钥
os.environ["DEEPSEEK_API_KEY"] = "sk-06810fb5453e4fd1b39e3e5f566da210"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_intent_recognition_comprehensive():
    """全面测试意图识别功能"""
    print("\n" + "="*60)
    print("🧠 全面测试意图识别功能")
    print("="*60)
    
    try:
        from intent_recognizer import get_intent_recognizer
        
        # 初始化意图识别器
        recognizer = get_intent_recognizer(llm_provider="deepseek")
        
        # 更全面的测试查询
        test_cases = [
            # 信息查询类
            ("什么是机器学习？", "信息查询"),
            ("人工智能的定义是什么？", "信息查询"),
            ("请介绍一下深度学习", "信息查询"),
            
            # 比较类问题
            ("深度学习和机器学习的区别是什么？", "比较类问题"),
            ("Python和Java哪个更好？", "比较类问题"),
            ("对比一下监督学习和无监督学习", "比较类问题"),
            
            # 深度解释类
            ("详细解释神经网络的工作原理", "深度解释"),
            ("请深入分析卷积神经网络的结构", "深度解释"),
            ("为什么要使用激活函数？", "深度解释"),
            
            # 推理分析类
            ("为什么深度学习在图像识别中效果更好？", "推理分析"),
            ("分析一下过拟合产生的原因", "推理分析"),
            ("预测人工智能的发展趋势", "推理分析"),
            
            # 操作指南类
            ("如何训练一个神经网络模型？", "操作指南"),
            ("怎样调试机器学习模型？", "操作指南"),
            ("如何选择合适的算法？", "操作指南"),
            
            # 个人观点类
            ("你认为哪种算法更适合这个场景？", "个人观点"),
            ("推荐一些学习资源", "个人观点"),
            ("你的建议是什么？", "个人观点"),
            
            # 闲聊类
            ("你好，今天天气怎么样？", "闲聊"),
            ("谢谢你的帮助", "闲聊"),
            ("再见", "闲聊"),
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for query, expected_intent in test_cases:
            print(f"\n📝 测试查询: {query}")
            print(f"   🎯 期望意图: {expected_intent}")
            
            try:
                result = recognizer.recognize_intent(query)
                predicted_intent = result['intent']
                confidence = result['confidence']
                
                print(f"   🤖 预测意图: {predicted_intent}")
                print(f"   📊 置信度: {confidence:.3f}")
                
                # 检查预测是否正确
                if predicted_intent == expected_intent:
                    print(f"   ✅ 预测正确")
                    correct_predictions += 1
                else:
                    print(f"   ❌ 预测错误")
                
                # 测试检索策略
                strategy = recognizer.get_retrieval_strategy(predicted_intent)
                print(f"   🎯 检索策略: top_k={strategy['top_k']}, vector_weight={strategy['vector_weight']}")
                
            except Exception as e:
                print(f"   ❌ 错误: {str(e)}")
        
        accuracy = correct_predictions / total_predictions
        print(f"\n📊 意图识别准确率: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
        
        # 如果准确率超过70%就认为测试通过
        success = accuracy >= 0.7
        if success:
            print("✅ 意图识别测试通过！")
        else:
            print("❌ 意图识别准确率偏低")
            
        return success
        
    except Exception as e:
        print(f"❌ 意图识别测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_qa_generation_comprehensive():
    """全面测试问答对生成功能"""
    print("\n" + "="*60)
    print("💬 全面测试问答对生成功能")
    print("="*60)
    
    try:
        from data_converter import DocumentConverter
        from langchain.docstore.document import Document
        
        # 初始化文档转换器
        converter = DocumentConverter(
            llm_provider="deepseek",
            api_key="sk-06810fb5453e4fd1b39e3e5f566da210"
        )
        
        # 测试不同类型的文本内容
        test_contents = [
            # 技术文档
            """
            机器学习算法分类
            
            监督学习：
            监督学习是一种机器学习方法，使用标记的训练数据来学习输入和输出之间的映射关系。
            常见算法包括线性回归、决策树、支持向量机等。
            
            无监督学习：
            无监督学习处理没有标签的数据，目标是发现数据中的隐藏模式。
            主要包括聚类、降维和关联规则学习等方法。
            """,
            
            # 概念解释
            """
            深度学习基础
            
            神经网络是深度学习的基础，由多个层次的神经元组成。
            每个神经元接收输入，通过权重和偏置进行计算，然后通过激活函数产生输出。
            反向传播算法用于训练神经网络，通过梯度下降优化权重参数。
            """,
            
            # 应用案例
            """
            计算机视觉应用
            
            图像分类：识别图像中的主要对象类别
            目标检测：定位并识别图像中的多个对象
            语义分割：为图像中的每个像素分配类别标签
            这些技术广泛应用于自动驾驶、医疗诊断、安防监控等领域。
            """
        ]
        
        total_qa_pairs = 0
        successful_generations = 0
        
        for i, content in enumerate(test_contents):
            print(f"\n📄 测试文本 {i+1}:")
            print(f"内容长度: {len(content)} 字符")
            
            # 创建文档对象
            docs = [Document(page_content=content, metadata={"source": f"test_text_{i+1}"})]
            
            # 分割文本
            chunks = converter.text_splitter.split_documents(docs)
            print(f"分割后文本块数量: {len(chunks)}")
            
            # 生成问答对
            start_time = time.time()
            qa_pairs = converter._generate_qa_pairs(chunks)
            gen_time = time.time() - start_time
            
            print(f"生成时间: {gen_time:.2f} 秒")
            print(f"生成问答对数量: {len(qa_pairs)}")
            
            if len(qa_pairs) > 0:
                successful_generations += 1
                total_qa_pairs += len(qa_pairs)
                
                # 显示生成的问答对
                print("生成的问答对:")
                for j, qa in enumerate(qa_pairs[:2]):  # 只显示前2个
                    print(f"  Q{j+1}: {qa.question}")
                    print(f"  A{j+1}: {qa.answer}")
                    print()
            else:
                print("❌ 未生成任何问答对")
        
        print(f"\n📊 问答对生成统计:")
        print(f"成功生成的文本数: {successful_generations}/{len(test_contents)}")
        print(f"总问答对数量: {total_qa_pairs}")
        print(f"平均每个文本生成: {total_qa_pairs/len(test_contents):.1f} 个问答对")
        
        # 如果至少有2/3的文本成功生成问答对，就认为测试通过
        success = successful_generations >= len(test_contents) * 2 / 3
        if success:
            print("✅ 问答对生成测试通过！")
        else:
            print("❌ 问答对生成成功率偏低")
            
        return success
        
    except Exception as e:
        print(f"❌ 问答对生成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """测试意图识别和问答对生成的集成"""
    print("\n" + "="*60)
    print("🔗 测试意图识别和问答对生成集成")
    print("="*60)
    
    try:
        from intent_recognizer import get_intent_recognizer
        from data_converter import DocumentConverter
        from langchain.docstore.document import Document
        
        # 初始化组件
        intent_recognizer = get_intent_recognizer("deepseek")
        converter = DocumentConverter(
            llm_provider="deepseek",
            api_key="sk-06810fb5453e4fd1b39e3e5f566da210"
        )
        
        # 测试场景：用户查询 + 文档处理
        user_query = "深度学习和传统机器学习有什么区别？"
        document_content = """
        机器学习技术对比
        
        传统机器学习：
        - 需要手工设计特征
        - 模型相对简单
        - 训练数据量要求较小
        - 可解释性较强
        
        深度学习：
        - 自动学习特征表示
        - 模型复杂，层次深
        - 需要大量训练数据
        - 在复杂任务上表现更好
        """
        
        print(f"🤔 用户查询: {user_query}")
        
        # 1. 意图识别
        intent_result = intent_recognizer.recognize_intent(user_query)
        print(f"🧠 识别意图: {intent_result['intent']} (置信度: {intent_result['confidence']:.3f})")
        
        # 2. 获取针对该意图的策略
        strategy = intent_recognizer.get_retrieval_strategy(intent_result['intent'])
        template = intent_recognizer.get_prompt_template(intent_result['intent'])
        print(f"🎯 检索策略: {strategy}")
        print(f"📝 提示模板: {template[:100]}...")
        
        # 3. 处理文档并生成问答对
        docs = [Document(page_content=document_content, metadata={"source": "comparison_doc"})]
        chunks = converter.text_splitter.split_documents(docs)
        qa_pairs = converter._generate_qa_pairs(chunks)
        
        print(f"📄 文档处理: {len(chunks)} 个文本块")
        print(f"💬 生成问答对: {len(qa_pairs)} 个")
        
        # 4. 显示结果
        if qa_pairs:
            print("\n生成的问答对:")
            for i, qa in enumerate(qa_pairs):
                print(f"{i+1}. Q: {qa.question}")
                print(f"   A: {qa.answer}")
                print()
        
        # 5. 模拟基于意图的回答生成
        context = document_content
        prompt = template.format(context=context, question=user_query)
        print(f"🤖 基于意图的提示词长度: {len(prompt)} 字符")
        
        success = len(qa_pairs) > 0 and intent_result['confidence'] > 0.5
        if success:
            print("✅ 集成测试通过！")
        else:
            print("❌ 集成测试失败")
            
        return success
        
    except Exception as e:
        print(f"❌ 集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始核心功能测试")
    print(f"📅 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试项目列表
    tests = [
        ("意图识别全面测试", test_intent_recognition_comprehensive),
        ("问答对生成全面测试", test_qa_generation_comprehensive),
        ("功能集成测试", test_integration),
    ]
    
    results = []
    start_time = time.time()
    
    # 执行所有测试
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {str(e)}")
            results.append((test_name, False))
    
    # 总结测试结果
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("📊 核心功能测试结果总结")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过 ({passed/total*100:.1f}%)")
    print(f"总耗时: {total_time:.2f} 秒")
    
    if passed == total:
        print("\n🎉 所有核心功能测试通过！")
        print("🌟 意图识别和问答对生成功能已完全修复！")
        print("🚀 系统核心功能运行正常，可以开始使用！")
    else:
        print(f"\n⚠️  {total-passed} 项测试失败，需要进一步检查")
        
    return passed == total

if __name__ == "__main__":
    main() 
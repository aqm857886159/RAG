#!/usr/bin/env python3
"""
DeepSeek API功能测试脚本
测试使用DeepSeek API的高级功能
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# 设置环境变量
os.environ['DEEPSEEK_API_KEY'] = 'sk-06810fb5453e4fd1b39e3e5f566da210'
os.environ['DEEPSEEK_API_BASE'] = 'https://api.deepseek.com'
os.environ['DEFAULT_LLM_PROVIDER'] = 'deepseek'

def test_deepseek_connection():
    """测试DeepSeek API连接"""
    print("🔗 测试DeepSeek API连接...")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(
            api_key="sk-06810fb5453e4fd1b39e3e5f566da210",
            base_url="https://api.deepseek.com"
        )
        
        # 简单的API调用测试
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "你好，请回复'API连接成功'"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        print(f"✅ DeepSeek API连接成功")
        print(f"   响应: {result}")
        return True
        
    except Exception as e:
        print(f"❌ DeepSeek API连接失败: {str(e)}")
        return False

def test_intent_recognition_with_api():
    """测试意图识别功能"""
    print("\n🧠 测试意图识别功能...")
    
    try:
        from intent_recognizer import IntentRecognizer
        recognizer = IntentRecognizer()
        
        test_queries = [
            "什么是机器学习？",
            "帮我总结一下这个文档的主要内容",
            "文档中提到了哪些关键技术？"
        ]
        
        success_count = 0
        for query in test_queries:
            print(f"查询: {query}")
            try:
                intent = recognizer.recognize_intent(query)
                print(f"✅ 意图识别成功: {intent}")
                success_count += 1
            except Exception as e:
                print(f"⚠️ 意图识别失败: {str(e)[:100]}...")
        
        print(f"意图识别测试完成: {success_count}/{len(test_queries)} 成功")
        return success_count > 0
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_qa_generation_with_api():
    """测试问答对生成功能"""
    print("\n📝 测试问答对生成功能...")
    
    try:
        from data_converter import DocumentConverter
        converter = DocumentConverter(llm_provider="deepseek")
        
        test_text = """
        机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。
        机器学习算法通过分析大量数据来识别模式，并使用这些模式来对新数据进行预测或决策。
        常见的机器学习类型包括监督学习、无监督学习和强化学习。
        """
        
        print(f"测试文本: {test_text[:50]}...")
        
        if converter.generator is not None:
            try:
                from langchain.docstore.document import Document
                test_doc = Document(page_content=test_text)
                qa_pairs = converter._generate_qa_pairs([test_doc], qa_per_chunk=2)
                
                print(f"✅ 问答对生成成功，生成了 {len(qa_pairs)} 对问答")
                for i, qa in enumerate(qa_pairs[:2]):
                    print(f"  Q{i+1}: {qa.question}")
                    print(f"  A{i+1}: {qa.answer}")
                    print()
                return True
                
            except Exception as e:
                print(f"⚠️ 问答对生成过程出错: {str(e)}")
                return False
        else:
            print("⚠️ 生成器未初始化")
            return False
            
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_document_conversion_with_api():
    """测试完整文档转换流程"""
    print("\n📄 测试完整文档转换流程...")
    
    try:
        from data_converter import DocumentConverter
        import tempfile
        
        # 创建测试文档
        test_content = """
        # 人工智能技术概述
        
        人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
        
        ## 主要技术领域
        
        1. **机器学习**: 使计算机能够从数据中学习，而无需明确编程
        2. **深度学习**: 基于人工神经网络的机器学习方法
        3. **自然语言处理**: 使计算机能够理解和生成人类语言
        4. **计算机视觉**: 使计算机能够解释和理解视觉信息
        
        ## 应用场景
        
        - 智能助手和聊天机器人
        - 图像识别和分析
        - 语音识别和合成
        - 推荐系统
        - 自动驾驶
        """
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name
        
        try:
            converter = DocumentConverter(llm_provider="deepseek")
            
            # 处理文档
            result = converter.process_document(temp_file, output_formats=["json", "qa"])
            
            print(f"✅ 文档转换成功")
            print(f"   文本块数量: {len(result.text_chunks)}")
            print(f"   问答对数量: {len(result.qa_pairs)}")
            print(f"   原始文本长度: {len(result.raw_text) if result.raw_text else 0} 字符")
            
            # 显示部分结果
            if result.qa_pairs:
                print("\n生成的问答对示例:")
                for i, qa in enumerate(result.qa_pairs[:2]):
                    print(f"  Q{i+1}: {qa.question}")
                    print(f"  A{i+1}: {qa.answer[:100]}...")
                    print()
            
            return True
            
        finally:
            # 清理临时文件
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"❌ 文档转换测试失败: {str(e)}")
        return False

def test_rag_qa_with_api():
    """测试RAG问答功能"""
    print("\n🤖 测试RAG问答功能...")
    
    try:
        from vectorizer import Vectorizer
        from retriever import get_retriever
        from generator import get_generator
        from langchain.docstore.document import Document
        
        # 创建测试文档
        test_docs = [
            Document(page_content="机器学习是人工智能的一个重要分支，它使计算机能够从数据中学习。"),
            Document(page_content="深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的工作方式。"),
            Document(page_content="自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言的交互。")
        ]
        
        # 创建向量存储
        vectorizer = Vectorizer()
        vector_store = vectorizer.create_vector_store(test_docs, index_name="test_api")
        
        if vector_store is not None:
            print("✅ 向量存储创建成功")
            
            # 创建检索器
            retriever = get_retriever(vector_store, use_hybrid=True, top_k=2)
            
            # 创建生成器
            generator = get_generator(provider="deepseek")
            
            # 测试问答
            query = "什么是机器学习？"
            print(f"查询: {query}")
            
            # 检索相关文档
            docs = retriever.retrieve(query)
            print(f"✅ 检索到 {len(docs)} 个相关文档")
            
            # 生成回答
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"""
            基于以下上下文信息回答问题：
            
            上下文：
            {context}
            
            问题：{query}
            
            请提供准确、简洁的回答：
            """
            
            response = generator.llm(prompt)
            print(f"✅ RAG问答成功")
            print(f"   回答: {response}")
            
            return True
        else:
            print("❌ 向量存储创建失败")
            return False
            
    except Exception as e:
        print(f"❌ RAG问答测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("🚀 DeepSeek API功能完整测试")
    print("=" * 60)
    
    # 测试结果记录
    results = {}
    
    # 1. 测试API连接
    results['API连接'] = test_deepseek_connection()
    
    # 2. 测试意图识别
    results['意图识别'] = test_intent_recognition_with_api()
    
    # 3. 测试问答对生成
    results['问答对生成'] = test_qa_generation_with_api()
    
    # 4. 测试文档转换
    results['文档转换'] = test_document_conversion_with_api()
    
    # 5. 测试RAG问答
    results['RAG问答'] = test_rag_qa_with_api()
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📊 DeepSeek API测试结果总结")
    print("=" * 60)
    
    success_count = 0
    total_count = len(results)
    
    for feature, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {feature}")
        if status:
            success_count += 1
    
    print(f"\n总体成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\n🎉 所有功能测试通过！DeepSeek API配置完全可用")
        print("✨ 现在可以使用完整的RAG智能问答和文档转换功能")
    elif success_count > 0:
        print(f"\n⚠️ 部分功能可用，{success_count}个功能测试成功")
        print("💡 建议检查失败的功能模块")
    else:
        print("\n❌ 所有功能测试失败，请检查API配置和网络连接")
    
    print("\n🎯 下一步建议:")
    print("1. 运行完整应用: python run_app.py")
    print("2. 测试Web界面的智能问答功能")
    print("3. 尝试文档转换和问答对生成")

if __name__ == "__main__":
    main() 
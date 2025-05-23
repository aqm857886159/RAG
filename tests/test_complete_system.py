#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整系统测试脚本
测试RAG系统的所有核心功能
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

def test_document_processing():
    """测试文档处理功能"""
    print("\n" + "="*60)
    print("📄 测试文档处理功能")
    print("="*60)
    
    try:
        from document_loader import DocumentLoader
        from text_splitter import get_text_splitter
        
        # 初始化组件
        loader = DocumentLoader()
        splitter = get_text_splitter("recursive", chunk_size=500, chunk_overlap=50)
        
        # 测试文档加载
        test_file = "docs/test_document.txt"
        if Path(test_file).exists():
            docs = loader.load_single_document(test_file)
            print(f"✅ 文档加载成功: {len(docs)} 个片段")
            
            # 测试文本分割
            chunks = splitter.split_documents(docs)
            print(f"✅ 文本分割成功: {len(chunks)} 个文本块")
            
            return True
        else:
            print(f"❌ 测试文档不存在: {test_file}")
            return False
            
    except Exception as e:
        print(f"❌ 文档处理测试失败: {str(e)}")
        return False

def test_vector_storage():
    """测试向量存储功能"""
    print("\n" + "="*60)
    print("🗄️ 测试向量存储功能")
    print("="*60)
    
    try:
        from vectorizer import get_vectorizer
        from document_loader import DocumentLoader
        from text_splitter import get_text_splitter
        
        # 初始化组件 - 修复参数传递
        vectorizer = get_vectorizer(
            embedding_model_name="BAAI/bge-small-zh-v1.5",
            vector_store_type="faiss"
        )
        loader = DocumentLoader()
        splitter = get_text_splitter("recursive", chunk_size=500, chunk_overlap=50)
        
        # 加载测试文档
        test_file = "docs/test_document.txt"
        if Path(test_file).exists():
            docs = loader.load_single_document(test_file)
            chunks = splitter.split_documents(docs)
            
            # 构建向量数据库
            start_time = time.time()
            vectorizer.add_documents(chunks)
            build_time = time.time() - start_time
            
            print(f"✅ 向量数据库构建成功: {len(chunks)} 个文档，耗时 {build_time:.2f} 秒")
            
            # 测试检索
            query = "什么是机器学习？"
            start_time = time.time()
            results = vectorizer.similarity_search(query, k=3)
            search_time = time.time() - start_time
            
            print(f"✅ 向量检索成功: 找到 {len(results)} 个相关文档，耗时 {search_time:.3f} 秒")
            
            return True
        else:
            print(f"❌ 测试文档不存在: {test_file}")
            return False
            
    except Exception as e:
        print(f"❌ 向量存储测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_retrieval_system():
    """测试检索系统功能"""
    print("\n" + "="*60)
    print("🔍 测试检索系统功能")
    print("="*60)
    
    try:
        from retriever import get_retriever
        from vectorizer import get_vectorizer
        from document_loader import DocumentLoader
        from text_splitter import get_text_splitter
        
        # 初始化组件 - 修复参数传递
        vectorizer = get_vectorizer(
            embedding_model_name="BAAI/bge-small-zh-v1.5",
            vector_store_type="faiss"
        )
        retriever = get_retriever("hybrid", vectorizer=vectorizer)
        loader = DocumentLoader()
        splitter = get_text_splitter("recursive", chunk_size=500, chunk_overlap=50)
        
        # 准备数据
        test_file = "docs/test_document.txt"
        if Path(test_file).exists():
            docs = loader.load_single_document(test_file)
            chunks = splitter.split_documents(docs)
            vectorizer.add_documents(chunks)
            
            # 测试检索
            query = "深度学习和机器学习的区别"
            start_time = time.time()
            results = retriever.get_relevant_documents(query, top_k=3)
            search_time = time.time() - start_time
            
            print(f"✅ 混合检索成功: 找到 {len(results)} 个相关文档，耗时 {search_time:.3f} 秒")
            
            # 显示检索结果
            for i, doc in enumerate(results[:2]):
                print(f"\n📄 结果 {i+1}: {doc.page_content[:100]}...")
            
            return True
        else:
            print(f"❌ 测试文档不存在: {test_file}")
            return False
            
    except Exception as e:
        print(f"❌ 检索系统测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_system():
    """测试生成系统功能"""
    print("\n" + "="*60)
    print("🤖 测试生成系统功能")
    print("="*60)
    
    try:
        from generator import get_generator
        
        # 初始化生成器
        generator = get_generator("deepseek", api_key="sk-06810fb5453e4fd1b39e3e5f566da210")
        
        # 测试简单生成
        prompt = "请简单解释什么是机器学习？"
        start_time = time.time()
        response = generator.llm(prompt)
        gen_time = time.time() - start_time
        
        print(f"✅ 文本生成成功，耗时 {gen_time:.2f} 秒")
        print(f"📝 生成内容: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成系统测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_rag_pipeline():
    """测试完整RAG流水线"""
    print("\n" + "="*60)
    print("🔄 测试完整RAG流水线")
    print("="*60)
    
    try:
        from document_loader import DocumentLoader
        from text_splitter import get_text_splitter
        from vectorizer import get_vectorizer
        from retriever import get_retriever
        from generator import get_generator
        from intent_recognizer import get_intent_recognizer
        
        # 初始化所有组件 - 修复参数传递
        loader = DocumentLoader()
        splitter = get_text_splitter("recursive", chunk_size=500, chunk_overlap=50)
        vectorizer = get_vectorizer(
            embedding_model_name="BAAI/bge-small-zh-v1.5",
            vector_store_type="faiss"
        )
        retriever = get_retriever("hybrid", vectorizer=vectorizer)
        generator = get_generator("deepseek", api_key="sk-06810fb5453e4fd1b39e3e5f566da210")
        intent_recognizer = get_intent_recognizer("deepseek")
        
        # 1. 文档处理
        test_file = "docs/test_document.txt"
        if not Path(test_file).exists():
            print(f"❌ 测试文档不存在: {test_file}")
            return False
            
        docs = loader.load_single_document(test_file)
        chunks = splitter.split_documents(docs)
        vectorizer.add_documents(chunks)
        print(f"✅ 文档处理完成: {len(chunks)} 个文本块")
        
        # 2. 测试查询
        query = "深度学习和机器学习有什么区别？"
        print(f"\n🤔 用户查询: {query}")
        
        # 3. 意图识别
        intent_result = intent_recognizer.recognize_intent(query)
        print(f"🧠 意图识别: {intent_result['intent']} (置信度: {intent_result['confidence']:.3f})")
        
        # 4. 检索相关文档
        strategy = intent_recognizer.get_retrieval_strategy(intent_result['intent'])
        relevant_docs = retriever.get_relevant_documents(query, top_k=strategy['top_k'])
        print(f"🔍 检索到 {len(relevant_docs)} 个相关文档")
        
        # 5. 构建上下文
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # 6. 生成回答
        template = intent_recognizer.get_prompt_template(intent_result['intent'])
        prompt = template.format(context=context, question=query)
        
        start_time = time.time()
        answer = generator.llm(prompt)
        gen_time = time.time() - start_time
        
        print(f"\n💡 RAG回答 (耗时 {gen_time:.2f} 秒):")
        print(f"{answer}")
        
        print(f"\n✅ 完整RAG流水线测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 完整RAG流水线测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始完整系统测试")
    print(f"📅 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试项目列表
    tests = [
        ("文档处理", test_document_processing),
        ("向量存储", test_vector_storage),
        ("检索系统", test_retrieval_system),
        ("生成系统", test_generation_system),
        ("完整RAG流水线", test_complete_rag_pipeline),
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
    print("📊 测试结果总结")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 项测试通过 ({passed/total*100:.1f}%)")
    print(f"总耗时: {total_time:.2f} 秒")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统运行正常！")
        print("🌟 RAG系统已完全修复并可以正常使用！")
    else:
        print(f"\n⚠️  {total-passed} 项测试失败，需要进一步检查")
        
    return passed == total

if __name__ == "__main__":
    main() 
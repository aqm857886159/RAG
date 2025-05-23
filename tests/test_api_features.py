#!/usr/bin/env python3
"""
API功能测试脚本
测试需要API配置的高级功能
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_env_config():
    """测试环境配置"""
    print("🔍 检查环境配置...")
    
    # 检查.env文件
    env_file = os.path.join(project_root, '.env')
    if os.path.exists(env_file):
        print(f"✅ .env文件存在: {env_file}")
        
        # 读取配置
        with open(env_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'OPENAI_API_KEY' in content:
            key_value = [line for line in content.split('\n') if 'OPENAI_API_KEY' in line]
            if key_value and not key_value[0].strip().endswith('='):
                print("✅ OpenAI API密钥已配置")
                return "openai"
        
        if 'DEEPSEEK_API_KEY' in content:
            key_value = [line for line in content.split('\n') if 'DEEPSEEK_API_KEY' in line]
            if key_value and not key_value[0].strip().endswith('='):
                print("✅ DeepSeek API密钥已配置")
                return "deepseek"
                
        print("⚠️ API密钥需要配置")
        return None
    else:
        print("❌ .env文件不存在")
        return None

def test_intent_recognition():
    """测试意图识别功能"""
    print("\n🧠 测试意图识别功能...")
    
    try:
        from intent_recognizer import IntentRecognizer
        recognizer = IntentRecognizer()
        
        test_queries = [
            "什么是机器学习？",
            "帮我总结一下这个文档",
            "文档中提到了哪些关键技术？"
        ]
        
        for query in test_queries:
            print(f"查询: {query}")
            try:
                intent = recognizer.recognize_intent(query)
                print(f"✅ 意图识别成功: {intent}")
                return True
            except Exception as e:
                print(f"⚠️ 意图识别需要API配置: {str(e)[:100]}...")
                return False
                
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_qa_generation():
    """测试问答对生成功能"""
    print("\n📝 测试问答对生成功能...")
    
    try:
        from data_converter import DocumentConverter
        converter = DocumentConverter()
        
        test_text = """
        机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。
        机器学习算法通过训练数据来构建数学模型，以便对新数据进行预测或决策。
        """
        
        print(f"测试文本: {test_text[:50]}...")
        try:
            if converter.generator is not None:
                from langchain.docstore.document import Document
                test_doc = Document(page_content=test_text)
                qa_pairs = converter._generate_qa_pairs([test_doc], qa_per_chunk=2)
                print(f"✅ 问答对生成成功，生成了 {len(qa_pairs)} 对问答")
                for i, qa in enumerate(qa_pairs[:2]):
                    print(f"  Q{i+1}: {qa.question[:50]}...")
                    print(f"  A{i+1}: {qa.answer[:50]}...")
                return True
            else:
                print("⚠️ 生成器未初始化，需要API配置")
                return False
        except Exception as e:
            print(f"⚠️ 问答对生成需要API配置: {str(e)[:100]}...")
            return False
            
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_retrieval_system():
    """测试检索系统"""
    print("\n🔍 测试检索系统...")
    
    try:
        from retriever import HybridRetriever
        from vectorizer import Vectorizer
        
        # 创建测试文档
        test_docs = [
            "机器学习是人工智能的重要分支。",
            "深度学习是机器学习的一个子领域。",
            "神经网络是深度学习的基础。"
        ]
        
        # 测试向量化
        print("测试向量存储...")
        vectorizer = Vectorizer()
        try:
            from langchain.docstore.document import Document
            documents = [Document(page_content=text) for text in test_docs]
            vector_store = vectorizer.create_vector_store(documents)
            if vector_store is not None:
                print("✅ 向量存储创建成功")
            else:
                print("⚠️ 向量存储创建失败")
        except Exception as e:
            print(f"⚠️ 向量存储需要模型下载: {str(e)[:100]}...")
        
        # 测试检索
        print("测试混合检索...")
        retriever = HybridRetriever()
        try:
            retriever.setup(test_docs)
            results = retriever.retrieve("什么是机器学习？", top_k=2)
            print(f"✅ 检索成功，返回 {len(results)} 个结果")
            return True
        except Exception as e:
            print(f"⚠️ 检索功能部分可用: {str(e)[:100]}...")
            return False
            
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def test_ocr_dependencies():
    """测试OCR依赖"""
    print("\n👁️ 测试OCR依赖...")
    
    ocr_packages = [
        ('paddlepaddle', 'paddle'),
        ('paddleocr', 'paddleocr'),
        ('PyMuPDF', 'fitz')
    ]
    
    available = []
    missing = []
    
    for package_name, import_name in ocr_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name} 已安装")
            available.append(package_name)
        except ImportError:
            print(f"❌ {package_name} 未安装")
            missing.append(package_name)
    
    if missing:
        print(f"\n安装命令: pip install {' '.join(missing)}")
    
    return len(available), len(missing)

def test_excel_processing():
    """测试Excel处理功能"""
    print("\n📊 测试Excel处理功能...")
    
    try:
        import pandas as pd
        import tempfile
        import os
        
        # 创建测试Excel文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            test_data = {
                '姓名': ['张三', '李四', '王五'],
                '年龄': [25, 30, 35],
                '部门': ['技术部', '销售部', '人事部']
            }
            df = pd.DataFrame(test_data)
            df.to_excel(tmp.name, index=False)
            excel_file = tmp.name
        
        # 测试加载Excel
        from document_loader import DocumentLoader
        loader = DocumentLoader()
        
        try:
            content = loader.load_single_document(excel_file)
            if content:
                print(f"✅ Excel加载成功，返回 {len(content)} 个文档对象")
                # 清理测试文件
                os.unlink(excel_file)
                return True
            else:
                print("⚠️ Excel加载返回空结果")
                os.unlink(excel_file)
                return False
            
        except Exception as e:
            print(f"⚠️ Excel处理需要完善: {str(e)[:100]}...")
            os.unlink(excel_file)
            return False
            
    except ImportError as e:
        print(f"❌ Excel处理依赖缺失: {e}")
        return False

def create_api_config_template():
    """创建API配置模板"""
    print("\n📝 创建API配置模板...")
    
    env_template = """# RAG项目API配置
# 复制此文件为 .env 并填入你的API密钥

# OpenAI配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# DeepSeek配置  
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 默认配置
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=BAAI/bge-large-zh
DEFAULT_RERANKER_MODEL=BAAI/bge-reranker-large

# 系统配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# OCR配置
USE_OCR=true
OCR_THREADS=4

# 安全配置
MAX_FILE_SIZE_MB=100
ALLOWED_FILE_TYPES=pdf,docx,doc,xlsx,xls,csv,txt
"""
    
    env_file = os.path.join(project_root, '.env-template')
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_template)
    
    print(f"✅ 配置模板已创建: {env_file}")
    print("请复制为 .env 文件并填入你的API密钥")

def main():
    """主测试函数"""
    print("🚀 RAG项目API功能测试")
    print("=" * 50)
    
    # 测试环境配置
    api_provider = test_env_config()
    
    # 测试各个功能模块
    results = {
        '意图识别': test_intent_recognition(),
        '问答生成': test_qa_generation(),
        '检索系统': test_retrieval_system(),
        'Excel处理': test_excel_processing()
    }
    
    # 测试OCR依赖
    ocr_available, ocr_missing = test_ocr_dependencies()
    
    # 生成配置模板
    if not api_provider:
        create_api_config_template()
    
    # 总结报告
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    print("=" * 50)
    
    print(f"API配置状态: {'✅ 已配置' if api_provider else '❌ 需要配置'}")
    print(f"OCR依赖状态: {ocr_available}/{ocr_available + ocr_missing} 已安装")
    
    print("\n功能模块测试结果:")
    for feature, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {feature}")
    
    # 提供下一步建议
    print("\n🎯 下一步建议:")
    
    if not api_provider:
        print("1. 配置API密钥:")
        print("   - 复制 .env-template 为 .env")
        print("   - 填入OpenAI或DeepSeek API密钥")
        print("   - 重新运行此测试")
    
    if ocr_missing > 0:
        print("2. 安装OCR依赖:")
        print("   pip install paddlepaddle paddleocr PyMuPDF")
    
    if api_provider and ocr_missing == 0:
        print("3. 运行完整功能测试:")
        print("   python tests/test_basic_functions.py")
        print("   python run_app.py")
    
    print("\n✨ 项目当前状态: 基础功能稳定，高级功能需要配置")

if __name__ == "__main__":
    main() 
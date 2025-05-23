"""
测试RAG系统的命令行脚本
"""
import logging
from src.converter import convert_file
from src.chunker import get_chunker
from src.vectorizer import get_vectorizer
from src.generator import get_generator
from langchain.schema.messages import HumanMessage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 步骤1: 转换文档
    print("1. 转换文档...")
    text = convert_file('test_document.txt')
    print(f"文档长度: {len(text)} 字符")
    
    # 步骤2: 分块
    print("\n2. 文本分块...")
    chunker = get_chunker()
    docs = chunker.create_documents(text, metadata={'source': 'test_document.txt'})
    print(f"分块结果: {len(docs)} 个文本块")
    
    # 步骤3: 向量化
    print("\n3. 向量化...")
    vectorizer = get_vectorizer()
    vector_store = vectorizer.create_vector_store(docs, index_name='test_cmd')
    print(f"向量存储类型: {type(vector_store).__name__}")
    
    # 步骤4: 测试问答
    print("\n4. 测试问答...")
    generator = get_generator(
        use_rag=True,
        vector_store=vector_store,
        provider="deepseek"
    )
    
    # 测试问题
    questions = [
        "RAG系统有哪些技术优势？",
        "RAG系统的核心组件有哪些？",
        "RAG技术未来的发展方向是什么？"
    ]
    
    for i, question in enumerate(questions):
        print(f"\n问题 {i+1}: {question}")
        response = generator.generate(question)
        
        # 提取答案和来源
        answer = response.get("answer", "")
        sources = response.get("source_documents", [])
        
        print(f"回答: {answer}")
        print(f"来源文档数量: {len(sources)}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 
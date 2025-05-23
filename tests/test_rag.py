"""
测试RAG系统的基本功能
"""
import os
import logging
from pathlib import Path
import tempfile

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_document():
    """创建测试文档"""
    logger.info("创建测试文档...")
    
    # 创建临时文本文件
    content = """
    # RAG系统测试文档
    
    这是一个用于测试检索增强生成(RAG)系统的测试文档。
    
    ## 背景知识
    
    检索增强生成(RAG)是一种结合了检索系统和生成式AI的技术。它通过以下步骤工作：
    1. 将文档分割成小块
    2. 将这些块转换为向量嵌入
    3. 存储在向量数据库中
    4. 在用户提问时，检索相关文档块
    5. 将检索到的文档块与用户问题一起发送给LLM生成回答
    
    ## 技术优势
    
    RAG系统具有以下优势：
    - 能够访问最新信息，而不受预训练数据限制
    - 可以引用具体信息源，提高可信度
    - 减少模型幻觉现象
    - 可以处理专业领域知识
    
    ## 应用场景
    
    RAG系统常见的应用场景包括：
    - 客户支持和聊天机器人
    - 知识库查询
    - 文档摘要和问答
    - 企业内部知识管理
    """
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name
    
    logger.info(f"测试文档已创建: {temp_path}")
    return temp_path

def test_document_processing(file_path):
    """测试文档处理功能"""
    try:
        from src.converter import convert_file
        from src.chunker import get_chunker
        
        logger.info(f"处理文档: {file_path}")
        
        # 转换文档为文本
        text = convert_file(file_path)
        logger.info(f"文档转换成功，文本长度: {len(text)}")
        
        # 分割文本
        chunker = get_chunker()
        docs = chunker.create_documents(text, metadata={"source": os.path.basename(file_path)})
        logger.info(f"文档分割成功，共 {len(docs)} 个片段")
        
        # 显示第一个片段
        if docs:
            logger.info(f"第一个片段内容: {docs[0].page_content[:100]}...")
        
        return docs
    except Exception as e:
        logger.error(f"文档处理失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_vectorization(docs):
    """测试向量化功能"""
    try:
        from src.vectorizer import get_vectorizer
        
        logger.info("测试向量化...")
        
        # 获取向量化器
        vectorizer = get_vectorizer()
        
        # 创建向量存储
        vector_store = vectorizer.create_vector_store(docs, index_name="test_index")
        logger.info(f"向量存储创建成功: {type(vector_store).__name__}")
        
        return vector_store
    except Exception as e:
        logger.error(f"向量化测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_rag_qa(vector_store):
    """测试RAG问答功能"""
    try:
        from src.generator import get_generator
        from langchain.schema.messages import HumanMessage
        
        logger.info("测试RAG问答...")
        
        # 创建生成器
        generator = get_generator(
            use_rag=True,
            vector_store=vector_store,
            provider="deepseek"
        )
        
        # 测试问题
        question = "RAG系统有哪些主要优势？"
        logger.info(f"测试问题: {question}")
        
        # 生成回答
        response = generator.generate(question)
        
        # 提取答案和来源
        answer = response.get("answer", "")
        sources = response.get("source_documents", [])
        
        logger.info(f"RAG回答: {answer[:100]}...")
        logger.info(f"来源文档数量: {len(sources)}")
        
        return response
    except Exception as e:
        logger.error(f"RAG问答测试失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def cleanup(file_path):
    """清理测试文件"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"已删除测试文件: {file_path}")
            
        # 清理测试向量存储
        vector_store_path = Path("vector_store/test_index")
        if vector_store_path.exists():
            import shutil
            shutil.rmtree(vector_store_path)
            logger.info(f"已删除测试向量存储: {vector_store_path}")
    except Exception as e:
        logger.error(f"清理失败: {str(e)}")

def main():
    """主函数"""
    logger.info("开始测试RAG系统...")
    
    # 创建测试文档
    file_path = create_test_document()
    
    try:
        # 测试文档处理
        docs = test_document_processing(file_path)
        if not docs:
            logger.error("文档处理测试失败")
            print("❌ 文档处理测试失败！请检查日志获取详细信息。")
            return
            
        # 测试向量化
        vector_store = test_vectorization(docs)
        if not vector_store:
            logger.error("向量化测试失败")
            print("❌ 向量化测试失败！请检查日志获取详细信息。")
            return
            
        # 测试RAG问答
        response = test_rag_qa(vector_store)
        if not response:
            logger.error("RAG问答测试失败")
            print("❌ RAG问答测试失败！请检查日志获取详细信息。")
            return
            
        # 测试成功
        logger.info("RAG系统测试成功！")
        print("\n✅ RAG系统测试成功！\n")
        
        # 打印结果
        answer = response.get("answer", "")
        print(f"问题: RAG系统有哪些主要优势？")
        print(f"回答: {answer}")
        
    finally:
        # 清理测试文件
        cleanup(file_path)

if __name__ == "__main__":
    main() 
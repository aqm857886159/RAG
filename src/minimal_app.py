"""
最小化的RAG应用，仅包含核心功能
"""
import os
import sys
import logging
import streamlit as st
import tempfile
from pathlib import Path

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.converter import convert_file
from src.chunker import get_chunker
from src.vectorizer import get_vectorizer, SimpleMemoryVectorStore
from src.generator import get_generator
from utils.helpers import format_source_documents

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置页面配置
st.set_page_config(
    page_title="RAG问答系统",
    page_icon="🤖",
    layout="wide"
)

# 初始化会话状态
if 'history' not in st.session_state:
    st.session_state.history = []

if 'generator' not in st.session_state:
    st.session_state.generator = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

def process_document(uploaded_file):
    """处理上传的文档"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # 转换文档为文本
        try:
            text = convert_file(tmp_path)
            logger.info(f"已转换文档: {uploaded_file.name}, 文本长度: {len(text)}")
        finally:
            # 删除临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        # 分割文本
        chunker = get_chunker()
        docs = chunker.create_documents(text, metadata={"source": uploaded_file.name})
        logger.info(f"已分割文档: {uploaded_file.name}, 片段数: {len(docs)}")
        
        # 创建向量存储
        vectorizer = get_vectorizer()
        
        # 创建新的向量存储
        vector_store = vectorizer.create_vector_store(docs)
        logger.info(f"创建了新的向量存储: {uploaded_file.name}")
        
        # 保存到会话状态
        st.session_state.vector_store = vector_store
        
        # 创建生成器
        st.session_state.generator = get_generator(
            use_rag=True,
            vector_store=vector_store,
            provider="deepseek"  # 默认使用DeepSeek
        )
        
        return True, "文档处理完成！"
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        return False, f"处理文档时出错: {str(e)}"

def main():
    # 标题
    st.title("简化版RAG问答系统")
    st.markdown("上传文档并提问，基于文档内容获取智能回答")
    
    # 侧边栏
    with st.sidebar:
        st.header("文档上传")
        
        # 文件上传组件
        uploaded_file = st.file_uploader("上传PDF、DOCX或TXT文件", type=["pdf", "docx", "txt"])
        
        if uploaded_file is not None:
            if st.button("处理文档"):
                with st.spinner("正在处理文档..."):
                    success, message = process_document(uploaded_file)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        st.divider()
        
        # 显示当前状态
        st.subheader("系统状态")
        if st.session_state.vector_store is not None:
            st.success("已加载知识库")
        else:
            st.warning("未加载知识库")
            st.info("请上传并处理文档")
        
        # 添加重置按钮
        if st.button("重置对话"):
            if st.session_state.generator:
                st.session_state.generator.reset_memory()
            st.session_state.history = []
            st.info("对话已重置")
    
    # 主界面
    
    # 显示聊天历史
    for i, (question, answer) in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
    
    # 用户输入
    user_question = st.chat_input("请输入您的问题")
    if user_question:
        # 显示用户问题
        with st.chat_message("user"):
            st.write(user_question)
        
        # 生成回答
        with st.chat_message("assistant"):
            if st.session_state.generator is None:
                st.error("请先上传并处理文档")
            else:
                with st.spinner("正在生成回答..."):
                    try:
                        # 生成回答
                        response = st.session_state.generator.generate(user_question)
                        
                        # 提取回答和源文档
                        answer = response.get("answer", "")
                        source_documents = response.get("source_documents", [])
                        
                        # 显示回答
                        st.write(answer)
                        
                        # 如果有源文档，显示
                        if source_documents:
                            with st.expander("查看来源文档"):
                                st.write(format_source_documents(source_documents))
                        
                        # 添加到聊天历史
                        st.session_state.history.append((user_question, answer))
                    except Exception as e:
                        st.error(f"生成回答时出错: {str(e)}")

if __name__ == "__main__":
    main() 
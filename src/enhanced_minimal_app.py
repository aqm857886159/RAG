"""
增强版的RAG应用，添加混合检索功能
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
from src.retriever import get_retriever, BM25Retriever
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
    page_title="增强版RAG问答系统",
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

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if 'documents' not in st.session_state:
    st.session_state.documents = []

if 'settings' not in st.session_state:
    st.session_state.settings = {
        'use_hybrid': True,
        'use_reranker': True,
        'top_k': 5,
        'vector_weight': 0.7,
        'bm25_weight': 0.3,
        'provider': 'deepseek'
    }

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
        
        # 保存文档到会话
        st.session_state.documents = docs
        
        # 创建向量存储
        vectorizer = get_vectorizer()
        vector_store = vectorizer.create_vector_store(docs)
        logger.info(f"创建了新的向量存储: {uploaded_file.name}")
        
        # 保存到会话状态
        st.session_state.vector_store = vector_store
        
        # 创建检索器（混合检索）
        settings = st.session_state.settings
        retriever = get_retriever(
            vector_store=vector_store,
            top_k=settings['top_k'],
            use_hybrid=settings['use_hybrid'],
            use_reranker=settings['use_reranker'],
            vector_weight=settings['vector_weight'],
            bm25_weight=settings['bm25_weight']
        )
        st.session_state.retriever = retriever
        
        # 创建生成器
        st.session_state.generator = get_generator(
            use_rag=True,
            vector_store=vector_store,
            retriever=retriever,
            provider=settings['provider']
        )
        
        return True, "文档处理完成！"
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        return False, f"处理文档时出错: {str(e)}"

def update_retriever():
    """更新检索器配置"""
    if st.session_state.vector_store is None or len(st.session_state.documents) == 0:
        return False, "请先上传并处理文档"
    
    try:
        settings = st.session_state.settings
        retriever = get_retriever(
            vector_store=st.session_state.vector_store,
            top_k=settings['top_k'],
            use_hybrid=settings['use_hybrid'],
            use_reranker=settings['use_reranker'],
            vector_weight=settings['vector_weight'],
            bm25_weight=settings['bm25_weight']
        )
        st.session_state.retriever = retriever
        
        # 更新生成器
        st.session_state.generator = get_generator(
            use_rag=True,
            vector_store=st.session_state.vector_store,
            retriever=retriever,
            provider=settings['provider']
        )
        
        return True, "检索配置已更新！"
    except Exception as e:
        logger.error(f"更新检索器时出错: {str(e)}")
        return False, f"更新检索器时出错: {str(e)}"

def main():
    # 标题
    st.title("增强版RAG问答系统")
    st.markdown("上传文档并提问，基于混合检索获取更精准的智能回答")
    
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
        
        # 检索配置
        st.header("检索配置")
        
        settings = st.session_state.settings
        
        # 检索模式
        settings['use_hybrid'] = st.checkbox("使用混合检索", value=settings['use_hybrid'], 
                                           help="结合向量检索和关键词检索获取更全面的结果")
        
        settings['use_reranker'] = st.checkbox("使用重排序", value=settings['use_reranker'],
                                             help="使用语义重排序提高结果相关性")
        
        settings['top_k'] = st.slider("检索结果数量", min_value=1, max_value=10, value=settings['top_k'])
        
        # 混合检索权重
        if settings['use_hybrid']:
            st.subheader("混合检索权重")
            settings['vector_weight'] = st.slider("向量检索权重", min_value=0.0, max_value=1.0, 
                                                 value=settings['vector_weight'], step=0.1)
            settings['bm25_weight'] = 1.0 - settings['vector_weight']
            st.text(f"BM25检索权重: {settings['bm25_weight']:.1f}")
        
        # 生成模型选择
        settings['provider'] = st.selectbox("选择模型", ["deepseek", "openai"], index=0)
        
        # 更新检索器按钮
        if st.button("更新检索配置"):
            success, message = update_retriever()
            if success:
                st.success(message)
            else:
                st.error(message)
        
        st.divider()
        
        # 显示当前状态
        st.subheader("系统状态")
        if st.session_state.vector_store is not None:
            st.success("已加载知识库")
            st.text(f"文档数量: {len(st.session_state.documents)}")
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
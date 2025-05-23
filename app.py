import os
import logging
import streamlit as st
import tempfile
from datetime import datetime
from pathlib import Path

from src.vectorizer import Vectorizer, get_vectorizer
from src.converter import convert_file
from src.chunker import get_chunker
from src.generator import get_generator
from src.llm import get_available_llm_providers
from utils.helpers import format_source_documents
from config.config import (
    EMBEDDING_MODEL, VECTOR_STORE_DIR, DEFAULT_LLM_PROVIDER,
    OPENAI_API_KEY, DEEPSEEK_API_KEY
)

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
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置CSS样式
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .stDeployButton {display:none;}
    .css-1offfwp {font-size: 14px !important;}
    header {visibility: hidden;}
    .main {padding-top: 0rem;}
</style>
""", unsafe_allow_html=True)

def show_response(response, history):
    """显示回答及其来源文档"""
    if not response:
        st.error("生成回答时出错")
        return
    
    # 提取回答和源文档
    answer = response.get("answer", "")
    source_documents = response.get("source_documents", [])
    is_rag_mode = response.get("is_rag_mode", False)
    
    # 显示回答
    st.write("### 回答：")
    st.write(answer)
    
    # 如果有源文档，显示
    if source_documents:
        with st.expander("查看来源文档"):
            st.write(format_source_documents(source_documents))
    
    # 添加到聊天历史
    history.append((st.session_state.user_question, answer))
    
def create_or_get_generator():
    """创建或获取生成器"""
    # 检查是否已有索引
    vector_store_dir = Path(VECTOR_STORE_DIR)
    has_index = False
    
    try:
        if vector_store_dir.exists():
            # 检查是否有任何索引文件
            for item in vector_store_dir.glob("**/*"):
                if item.is_file() and item.suffix in ['.faiss', '.pkl', '.sqlite3']:
                    has_index = True
                    break
            
            # 如果目录存在但没有索引文件
            if not has_index:
                logger.warning(f"向量存储目录存在，但没有找到索引文件: {vector_store_dir}")
        else:
            logger.warning(f"向量存储目录不存在: {vector_store_dir}")
    except Exception as e:
        logger.error(f"检查向量存储目录时出错: {str(e)}")
    
    # 获取向量存储
    vector_store = None
    if has_index:
        try:
            # 尝试加载向量存储
            vectorizer = get_vectorizer()
            vector_store = vectorizer.load_vector_store(fallback_to_empty=True)
            logger.info("已加载向量存储")
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
    
    # 创建生成器
    try:
        # 检查配置的LLM提供商是否有API密钥
        providers = get_available_llm_providers()
        provider = DEFAULT_LLM_PROVIDER
        
        # 如果默认提供商不可用，尝试使用另一个
        if not providers.get(provider, False):
            for p, available in providers.items():
                if available:
                    provider = p
                    logger.info(f"默认提供商不可用，使用: {provider}")
                    break
        
        # 创建生成器
        generator = get_generator(
            use_rag=has_index and vector_store is not None,
            vector_store=vector_store,
            provider=provider
        )
        logger.info(f"已创建生成器: 使用RAG={has_index and vector_store is not None}, 提供商={provider}")
        return generator, has_index
    except Exception as e:
        logger.error(f"创建生成器失败: {str(e)}")
        st.error(f"初始化生成器失败: {str(e)}")
        # 尝试创建一个简单的生成器作为备选
        try:
            generator = get_generator(use_rag=False)
            logger.info("创建了备选简单生成器")
            return generator, False
        except Exception as inner_e:
            logger.error(f"创建备选生成器也失败: {str(inner_e)}")
            st.error(f"初始化备选生成器也失败: {str(inner_e)}")
            return None, False

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
        
        # 尝试加载已有的向量存储
        vector_store = vectorizer.load_vector_store(fallback_to_empty=True)
        
        # 如果加载失败或不存在，创建新的
        if vector_store is None:
            vector_store = vectorizer.create_vector_store(docs)
            logger.info(f"创建了新的向量存储: {uploaded_file.name}")
        else:
            # 向已有的向量存储添加文档
            vector_store = vectorizer.add_documents(docs, vector_store)
            logger.info(f"添加文档到现有向量存储: {uploaded_file.name}")
        
        # 刷新生成器
        st.session_state.generator, st.session_state.has_index = create_or_get_generator()
        
        return True, "文档处理完成！"
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        return False, f"处理文档时出错: {str(e)}"

def main():
    # 标题
    st.title("RAG问答系统")
    
    # 初始化会话状态
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    
    if 'generator' not in st.session_state or 'has_index' not in st.session_state:
        st.session_state.generator, st.session_state.has_index = create_or_get_generator()
    
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
        if st.session_state.has_index:
            st.success("已加载知识库")
        else:
            st.warning("未加载知识库或知识库为空")
            st.info("请上传并处理文档以启用RAG功能")
        
        # 显示LLM提供商状态
        providers = get_available_llm_providers()
        st.subheader("LLM提供商")
        for provider, available in providers.items():
            if available:
                st.success(f"✅ {provider.capitalize()}")
            else:
                st.error(f"❌ {provider.capitalize()} (API密钥未配置)")
        
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
    
    # 输入框
    user_question = st.chat_input("输入你的问题...")
    
    if user_question:
        # 保存问题
        st.session_state.user_question = user_question
        
        # 显示用户问题
        with st.chat_message("user"):
            st.write(user_question)
        
        # 显示助手正在输入
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                if st.session_state.generator:
                    try:
                        # 生成回答
                        response = st.session_state.generator.generate(user_question)
                        
                        # 显示回答
                        show_response(response, st.session_state.history)
                    except Exception as e:
                        logger.error(f"生成回答时出错: {str(e)}")
                        st.error(f"生成回答时出错: {str(e)}")
                else:
                    st.error("生成器未初始化，请检查日志查看错误信息")

if __name__ == "__main__":
    main() 
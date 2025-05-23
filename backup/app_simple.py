"""
简化版RAG应用，解决Windows兼容性问题
"""
import os
import logging
import streamlit as st
import tempfile
from pathlib import Path

from src.llm import get_llm, get_available_llm_providers
from config.config import EMBEDDING_MODEL, VECTOR_STORE_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_simple.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置页面配置
st.set_page_config(
    page_title="RAG问答系统 (简化版)",
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

def main():
    # 标题
    st.title("RAG问答系统 (简化版)")
    st.info("此版本是解决Windows兼容性问题的简化版。请使用app.py运行完整版。")
    
    # 初始化会话状态
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""
    
    # 侧边栏
    with st.sidebar:
        st.header("文档上传")
        
        # 文件上传组件
        uploaded_file = st.file_uploader("上传PDF、DOCX或TXT文件", type=["pdf", "docx", "txt"])
        
        if uploaded_file is not None:
            if st.button("处理文档"):
                st.warning("简化版不支持文档处理功能")
        
        st.divider()
        
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
                try:
                    # 尝试获取LLM
                    llm = get_llm()
                    
                    # 直接使用LLM生成回答
                    result = llm.predict_messages(
                        messages=[{"role": "user", "content": user_question}]
                    )
                    
                    # 提取回答文本
                    answer = result.content
                    
                    # 显示回答
                    st.write(answer)
                    
                    # 添加到聊天历史
                    st.session_state.history.append((user_question, answer))
                except Exception as e:
                    logger.error(f"生成回答时出错: {str(e)}")
                    st.error(f"生成回答时出错: {str(e)}")

if __name__ == "__main__":
    main() 
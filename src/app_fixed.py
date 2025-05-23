"""
Web应用主程序，提供友好的用户交互界面
"""
import os
import sys
import logging
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import pandas as pd
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.docstore.document import Document

from document_loader import DocumentLoader
from text_splitter import get_text_splitter
from vectorizer import get_vectorizer
from retriever import get_retriever
from generator import get_generator
from evaluator import get_evaluator, RAGEvaluator
from data_converter import DocumentConverter, StructuredOutput
from utils.helpers import time_function, ensure_directory, format_source_documents
from config.config import (
    DATA_DIR, VECTOR_STORE_DIR, EVAL_DATA_DIR, EVAL_RESULTS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, TOP_K, 
    OPENAI_API_KEY, OPENAI_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    DEFAULT_LLM_PROVIDER, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    DEFAULT_EVAL_DATASET,
    APP_TITLE, APP_DESCRIPTION,
    USE_HYBRID, USE_RERANKER,
    VECTOR_WEIGHT, BM25_WEIGHT
)
from models.deepseek_llm import DeepSeekLLM

# 将意图识别默认设置为False
USE_INTENT_RECOGNITION = False
INTENT_MODEL = "bert-base-chinese"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置页面标题
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔍",
    layout="wide"
)

# 页面标题
st.title(APP_TITLE)
st.markdown(APP_DESCRIPTION)

# 侧边栏配置
st.sidebar.title("配置")

# 确保目录存在
ensure_directory(DATA_DIR)
ensure_directory(VECTOR_STORE_DIR)
ensure_directory(EVAL_DATA_DIR)
ensure_directory(EVAL_RESULTS_DIR)

# 功能选择
app_mode = st.sidebar.selectbox(
    "选择功能",
    ["RAG问答系统", "文档结构化转换"],
    index=0
)

# 模型提供商选择
model_provider = st.sidebar.selectbox(
    "选择模型提供商",
    options=["OpenAI", "DeepSeek"],
    index=0 if DEFAULT_LLM_PROVIDER == "openai" else 1
)

# API Key输入
if model_provider == "OpenAI":
    api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password"
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
    # 模型选择
    model_name = st.sidebar.selectbox(
        "选择模型",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
else:  # DeepSeek
    api_key = st.sidebar.text_input(
        "DeepSeek API Key", 
        value=os.getenv("DEEPSEEK_API_KEY", ""),
        type="password"
    )
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        
    # 模型选择
    model_name = st.sidebar.selectbox(
        "选择模型",
        options=["deepseek-chat", "deepseek-coder"],
        index=0
    )

# 温度参数
temperature = st.sidebar.slider(
    "生成温度",
    min_value=0.0,
    max_value=1.0,
    value=LLM_TEMPERATURE,
    step=0.1,
    help="较低的值使输出更确定，较高的值使输出更随机"
)

# 最大生成长度
max_tokens = st.sidebar.slider(
    "最大生成长度",
    min_value=256,
    max_value=4096,
    value=LLM_MAX_TOKENS,
    step=256,
    help="限制生成回答的最大token数量"
)

# RAG问答系统模式
if app_mode == "RAG问答系统":
    # 检索数量
    top_k = st.sidebar.slider(
        "检索结果数量",
        min_value=1,
        max_value=10,
        value=TOP_K,
        step=1,
        help="检索的文档片段数量"
    )

    # 是否使用记忆
    use_memory = st.sidebar.checkbox(
        "启用对话记忆",
        value=True,
        help="记住对话历史以提供更连贯的回答"
    )

    # 是否使用意图识别
    use_intent_recognition = st.sidebar.checkbox(
        "启用意图识别",
        value=USE_INTENT_RECOGNITION,
        help="自动识别用户问题意图，优化检索和生成策略"
    )

    # 高级检索设置
    with st.sidebar.expander("高级检索设置", expanded=False):
        use_hybrid = st.checkbox(
            "启用混合检索",
            value=USE_HYBRID,
            help="结合向量检索和关键词检索"
        )
        
        use_reranker = st.checkbox(
            "启用重排序",
            value=USE_RERANKER,
            help="使用Cross-Encoder对检索结果重排序"
        )
        
        vector_weight = st.slider(
            "向量检索权重",
            min_value=0.0,
            max_value=1.0,
            value=VECTOR_WEIGHT,
            step=0.1,
            help="混合检索中向量检索的权重"
        )
        
        bm25_weight = st.slider(
            "BM25检索权重",
            min_value=0.0,
            max_value=1.0,
            value=BM25_WEIGHT,
            step=0.1,
            help="混合检索中BM25检索的权重"
        )

    # 向量存储类型
    vector_store_type = st.sidebar.selectbox(
        "向量存储类型",
        options=["FAISS", "Chroma"],
        index=0
    )

    # 意图识别设置
    with st.sidebar.expander("意图识别设置", expanded=False):
        if use_intent_recognition:
            intent_model = st.selectbox(
                "意图识别模型",
                options=["bert-base-chinese", "roberta-base-chinese"],
                index=0,
                help="用于识别用户意图的预训练模型"
            )
            
    # 设置会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
        
    if "generator" not in st.session_state:
        st.session_state.generator = None
        
    if "intent_recognizer" not in st.session_state:
        st.session_state.intent_recognizer = None

    # 选项卡：聊天、知识库管理、评估
    tab1, tab2, tab3 = st.tabs(["💬 聊天", "📚 知识库管理", "📊 系统评估"])

    # 聊天选项卡
    with tab1:
        # 显示聊天历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("source_documents"):
                    with st.expander("查看引用来源"):
                        st.markdown(format_source_documents(message["source_documents"]))
        
        # 创建检索器（如果尚未创建）
        if not st.session_state.retriever:
            try:
                # 初始化向量化器
                vectorizer = get_vectorizer(
                    embedding_model_name=EMBEDDING_MODEL,
                    vector_store_dir=VECTOR_STORE_DIR,
                    vector_store_type=vector_store_type.lower()
                )
                
                # 加载向量存储
                vector_store = vectorizer.load_vector_store()
                
                # 初始化检索器
                st.session_state.retriever = get_retriever(
                    vector_store=vector_store,
                    top_k=top_k,
                    use_hybrid=use_hybrid,
                    use_reranker=use_reranker,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight
                )
            except Exception as e:
                st.error(f"初始化检索器失败: {str(e)}")
                st.warning("请确保已上传并处理了知识库文档，或检查配置是否正确。")
                
        # 创建生成器（如果尚未创建）
        if not st.session_state.generator:
            try:
                st.session_state.generator = get_generator(
                    retriever=st.session_state.retriever,
                    model_name=model_name,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_memory=use_memory,
                    provider=model_provider.lower()
                )
            except Exception as e:
                st.error(f"初始化生成器失败: {str(e)}")
                st.warning("请检查API密钥是否正确配置。")
                
        # 创建意图识别器（如果启用且尚未创建）
        if use_intent_recognition and not st.session_state.intent_recognizer:
            try:
                from intent_recognizer import IntentRecognizer
                st.session_state.intent_recognizer = IntentRecognizer(
                    model_name=intent_model or INTENT_MODEL
                )
            except Exception as e:
                st.error(f"初始化意图识别器失败: {str(e)}")
                st.warning("请检查模型配置是否正确。")
                use_intent_recognition = False
        
        # 用户输入
        if prompt := st.chat_input("请输入您的问题"):
            # 添加用户消息到历史
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 显示助手消息（带加载动画）
            with st.chat_message("assistant"):
                if st.session_state.retriever and st.session_state.generator:
                    try:
                        message_placeholder = st.empty()
                        
                        # 识别意图（如果启用）
                        intent_type = None
                        if use_intent_recognition and st.session_state.intent_recognizer:
                            with st.spinner("正在分析问题意图..."):
                                intent_type = st.session_state.intent_recognizer.predict_intent(prompt)
                                st.session_state.messages[-1]["intent"] = intent_type
                                
                        # 执行检索
                        with st.spinner("正在检索相关文档..."):
                            search_results = st.session_state.retriever.search(
                                query=prompt,
                                top_k=top_k,
                                intent_type=intent_type
                            )
                            
                        # 生成回答
                        with st.spinner("正在生成回答..."):
                            response = st.session_state.generator.generate_answer(
                                query=prompt,
                                search_results=search_results,
                                intent_type=intent_type
                            )
                            
                        # 显示回答
                        message_placeholder.markdown(response.answer)
                        
                        # 显示引用来源
                        if response.source_documents:
                            with st.expander("查看引用来源"):
                                st.markdown(format_source_documents(response.source_documents))
                                
                        # 添加回答到历史
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response.answer,
                            "source_documents": response.source_documents
                        })
            
                    except Exception as e:
                        err_msg = f"处理请求时出错: {str(e)}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                else:
                    err_msg = "系统未准备就绪，请检查配置并确保已加载知识库"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

    # 知识库管理选项卡
    with tab2:
        st.header("知识库管理")
        
        # 文件上传区域
        uploaded_files = st.file_uploader(
            "上传文档到知识库", 
            type=["pdf", "docx", "doc", "txt", "csv"],
            accept_multiple_files=True
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chunk_size_upload = st.number_input(
                "文本块大小",
                min_value=100,
                max_value=2000,
                value=CHUNK_SIZE,
                step=50
            )
            
        with col2:
            chunk_overlap_upload = st.number_input(
                "块重叠大小",
                min_value=0,
                max_value=500,
                value=CHUNK_OVERLAP,
                step=10
            )
            
        with col3:
            embedding_model_upload = st.selectbox(
                "向量模型",
                options=["BAAI/bge-large-zh", "BAAI/bge-small-zh", "text2vec-large-chinese"],
                index=0
            )
        
        # 处理上传文件
        if uploaded_files and st.button("处理并索引文档"):
            with st.spinner("正在处理文档..."):
                # 保存上传文件
                saved_files = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        saved_files.append(file_path)
                
                st.success(f"已保存 {len(saved_files)} 个文件到数据目录")
                
                # 加载和处理文档
                try:
                    # 初始化文档加载器
                    loader = DocumentLoader(DATA_DIR)
                
                    # 加载文档
                    documents = loader.load_documents(saved_files)
                    st.info(f"已加载 {len(documents)} 个文档片段")
                
                    # 文本分块
                    text_splitter = get_text_splitter(
                        splitter_type="recursive",
                        chunk_size=chunk_size_upload,
                        chunk_overlap=chunk_overlap_upload
                    )
                    chunks = text_splitter.split_documents(documents)
                    st.info(f"已分割为 {len(chunks)} 个文本块")
                
                    # 初始化向量化器
                    vectorizer = get_vectorizer(
                        embedding_model_name=embedding_model_upload,
                        vector_store_dir=VECTOR_STORE_DIR,
                        vector_store_type=vector_store_type.lower()
                    )
                
                    # 创建或更新向量存储
                    vector_store = vectorizer.load_vector_store()
                    if vector_store is not None:
                        # 添加到现有向量存储
                        vector_store = vectorizer.add_documents(chunks, vector_store)
                    else:
                        # 创建新的向量存储
                        vector_store = vectorizer.create_vector_store(chunks)
                    
                    # 更新检索器
                    if st.session_state.retriever is not None:
                        st.session_state.retriever.update_vector_store(vector_store)
                    else:
                        # 创建新的检索器
                        st.session_state.retriever = get_retriever(
                            vector_store=vector_store,
                            top_k=top_k,
                            use_hybrid=use_hybrid,
                            use_reranker=use_reranker,
                            vector_weight=vector_weight,
                            bm25_weight=bm25_weight
                        )
                    
                    st.success(f"成功处理并索引 {len(chunks)} 个文本块")
                
                except Exception as e:
                    st.error(f"处理文档时出错: {str(e)}")
        
        # 知识库统计信息
        st.subheader("知识库统计")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(DATA_DIR):
                data_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
                st.metric("数据文件数量", len(data_files))
            else:
                st.metric("数据文件数量", 0)
        
        with col2:
            if st.session_state.retriever:
                try:
                    doc_count = st.session_state.retriever.get_document_count()
                    st.metric("索引文档数量", doc_count)
                except:
                    st.metric("索引文档数量", "未知")
            else:
                st.metric("索引文档数量", 0)
            
        # 显示数据目录文件列表
        if os.path.exists(DATA_DIR):
            with st.expander("查看数据文件"):
                data_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
                for file in data_files:
                    st.text(file)

    # 评估选项卡
    with tab3:
        st.header("系统评估")
        
        # 加载评估数据集
        eval_datasets = [f for f in os.listdir(EVAL_DATA_DIR) if f.endswith('.json')]
        
        if not eval_datasets:
            st.warning("未找到评估数据集，请上传至评估数据目录")
        else:
            selected_dataset = st.selectbox(
                "选择评估数据集",
                options=eval_datasets,
                index=0
            )
        
            # 评估配置
            col1, col2 = st.columns(2)
            
            with col1:
                eval_retriever = st.checkbox("评估检索质量", value=True)
                eval_generation = st.checkbox("评估生成质量", value=True)
                
            with col2:
                num_samples = st.number_input(
                    "评估样本数量",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1,
                    help="限制评估的问题数量"
                )
            
            # 执行评估
            if st.button("开始评估"):
                if not st.session_state.retriever or not st.session_state.generator:
                    st.error("请先确保检索器和生成器已初始化")
                else:
                    try:
                        with st.spinner("正在进行评估..."):
                            # 初始化评估器
                            evaluator = get_evaluator(
                                provider=model_provider.lower(),
                                model_name=model_name
                            )
                            
                            # 加载数据集
                            dataset_path = os.path.join(EVAL_DATA_DIR, selected_dataset)
                            
                            # 执行评估
                            eval_results = evaluator.evaluate(
                                dataset_path=dataset_path,
                                retriever=st.session_state.retriever,
                                generator=st.session_state.generator,
                                eval_retrieval=eval_retriever,
                                eval_generation=eval_generation,
                                num_samples=num_samples
                            )
                            
                            # 生成唯一的结果文件名
                            timestamp = int(time.time())
                            result_file = f"eval_result_{timestamp}.json"
                            result_path = os.path.join(EVAL_RESULTS_DIR, result_file)
                            
                            # 保存结果
                            with open(result_path, 'w', encoding='utf-8') as f:
                                json.dump(eval_results, f, ensure_ascii=False, indent=2)
                                
                            st.success(f"评估完成，结果已保存至: {result_file}")
                            
                            # 显示结果摘要
                            if eval_retriever and "retrieval" in eval_results:
                                st.subheader("检索评估结果")
                                retrieval_metrics = eval_results["retrieval"]["metrics"]
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("精确率 (Precision)", f"{retrieval_metrics.get('precision', 0):.3f}")
                                    
                                with col2:
                                    st.metric("召回率 (Recall)", f"{retrieval_metrics.get('recall', 0):.3f}")
                                    
                                with col3:
                                    st.metric("F1分数", f"{retrieval_metrics.get('f1', 0):.3f}")
                                    
                            if eval_generation and "generation" in eval_results:
                                st.subheader("生成评估结果")
                                generation_metrics = eval_results["generation"]["metrics"]
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("相关性得分", f"{generation_metrics.get('relevance', 0):.3f}")
                                    
                                with col2:
                                    st.metric("忠实度得分", f"{generation_metrics.get('faithfulness', 0):.3f}")
                                    
                            # 详细结果查看
                            with st.expander("查看详细评估结果"):
                                st.json(eval_results)
                                
                    except Exception as e:
                        st.error(f"评估过程中出错: {str(e)}")
                        
        # 查看历史评估结果
        st.subheader("历史评估结果")
        
        if os.path.exists(EVAL_RESULTS_DIR):
            result_files = [f for f in os.listdir(EVAL_RESULTS_DIR) if f.endswith('.json')]
            
            if not result_files:
                st.info("暂无历史评估结果")
            else:
                selected_result = st.selectbox(
                    "选择查看结果",
                    options=result_files,
                    index=0
                )
                
                if selected_result:
                    result_path = os.path.join(EVAL_RESULTS_DIR, selected_result)
                    
                    try:
                        with open(result_path, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                        
                        with st.expander("查看结果详情"):
                            st.json(result_data)
                            
                    except Exception as e:
                        st.error(f"加载结果文件时出错: {str(e)}")
        else:
            st.info("评估结果目录不存在")

# 文档结构化转换模式
elif app_mode == "文档结构化转换":
    st.header("文档结构化转换")
    st.markdown("将多种格式的文档转换为AI友好的结构化数据格式，包括JSON、问答对等")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["pdf", "docx", "doc", "txt", "xlsx", "xls", "csv"],
        help="支持PDF、Word、Excel和纯文本文件"
    )
    
    # 转换选项
    with st.expander("转换选项", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            output_formats = st.multiselect(
                "选择输出格式",
                options=["JSON", "JSONL", "问答对", "文本块"],
                default=["JSON", "问答对"]
            )
            
            use_ocr = st.checkbox(
                "使用OCR处理扫描文档",
                value=True,
                help="对PDF等文档使用光学字符识别技术提取文本"
            )
        
        with col2:
            chunk_size = st.slider(
                "文本块大小",
                min_value=100,
                max_value=2000,
                value=CHUNK_SIZE,
                step=50
            )
            
            chunk_overlap = st.slider(
                "块重叠大小",
                min_value=0,
                max_value=500,
                value=CHUNK_OVERLAP,
                step=10
            )
            
            qa_per_chunk = st.slider(
                "每块问答对数量",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                help="为每个文本块生成的问答对数量"
            )
    
    # 处理按钮
    if uploaded_file and st.button("开始处理"):
        with st.spinner("正在处理文档..."):
            # 保存上传的文件
            temp_file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 初始化转换器
            converter = DocumentConverter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                llm_provider=model_provider.lower(),
                temperature=temperature,
                use_ocr=use_ocr,
                api_key=api_key
            )
            
            # 处理文档
            try:
                result = converter.process_document(
                    temp_file_path,
                    output_formats=[f.lower() for f in output_formats]
                )
                
                # 显示结果
                st.success(f"文档处理完成！共生成 {len(result.text_chunks)} 个文本块。")
                
                # 显示文本块
                if "文本块" in output_formats:
                    with st.expander("文本块预览", expanded=False):
                        for i, chunk in enumerate(result.text_chunks[:5]):  # 只显示前5个
                            st.markdown(f"**块 {i+1}**")
                            st.text(chunk["content"][:500] + "..." if len(chunk["content"]) > 500 else chunk["content"])
                            st.json(chunk["metadata"])
                        if len(result.text_chunks) > 5:
                            st.info(f"还有 {len(result.text_chunks) - 5} 个文本块未显示")
            
                # 显示问答对
                if "问答对" in output_formats and result.qa_pairs:
                    with st.expander("问答对预览", expanded=True):
                        for i, qa in enumerate(result.qa_pairs[:10]):  # 只显示前10个
                            st.markdown(f"**问题 {i+1}**: {qa.question}")
                            st.markdown(f"**回答**: {qa.answer}")
                            st.markdown("---")
                
                # 显示表格数据
                if result.tables:
                    with st.expander("表格数据预览", expanded=True):
                        for i, table in enumerate(result.tables[:3]):  # 只显示前3个表格
                            st.markdown(f"**表格 {i+1}** (来自: {table.get('sheet_name', '未知工作表')})")
                            
                            if 'data' in table and table['data']:
                                df = pd.DataFrame(table['data'])
                                st.dataframe(df)
                            else:
                                st.info("表格无数据或数据为空")
                            
                            st.markdown("---")
                
                # 提供下载
                st.subheader("下载转换结果")
                
                # JSON下载
                if "JSON" in output_formats:
                    json_str = result.model_dump_json(indent=2)
                    st.download_button(
                        label="下载完整JSON文件",
                        data=json_str.encode("utf-8"),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_structured.json",
                        mime="application/json"
                    )
                
                # JSONL下载
                if "JSONL" in output_formats:
                    jsonl_str = converter.convert_to_format(result, "jsonl")
                    st.download_button(
                        label="下载JSONL文件",
                        data=jsonl_str.encode("utf-8"),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_chunks.jsonl",
                        mime="application/jsonl"
                    )
                
                # 问答对下载
                if "问答对" in output_formats and result.qa_pairs:
                    qa_json = converter.convert_to_format(result, "qa_json")
                    st.download_button(
                        label="下载问答对JSON文件",
                        data=qa_json.encode("utf-8"),
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_qa_pairs.json",
                        mime="application/json"
                    )
                    
            except Exception as e:
                st.error(f"处理文档时发生错误: {str(e)}")
                logging.error(f"处理文档失败: {temp_file_path}, 错误: {str(e)}")
            
            # 清理临时文件（可选）
            # os.remove(temp_file_path)
    
    # 说明和示例
    with st.expander("功能说明和使用示例", expanded=False):
        st.markdown("""
        ### 文档结构化转换功能说明

        此功能可以将各种格式的文档转换为结构化数据，便于AI应用使用。

        #### 支持的文档格式
        - **PDF文件**：支持文本PDF和扫描PDF（使用OCR）
        - **Word文档**：支持.docx和.doc格式
        - **Excel表格**：支持.xlsx和.xls格式
        - **纯文本文件**：支持.txt格式
        - **CSV文件**：支持.csv格式

        #### 输出格式说明
        - **JSON**：完整的结构化数据，包含文本块、问答对和元数据
        - **JSONL**：每行一个JSON对象，便于流式处理
        - **问答对**：从文档内容自动生成的问题和答案对
        - **文本块**：分割后的文本片段，带有元数据

        #### 使用建议
        - 对于扫描PDF，请启用OCR选项
        - 文本块大小建议在300-500之间，以获得最佳的语义完整性
        - 生成问答对可能需要较长时间，特别是对于大型文档
        """)

if __name__ == "__main__":
    pass 
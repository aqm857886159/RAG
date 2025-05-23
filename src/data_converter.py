"""
数据转换模块，负责将各种格式的文档转换为AI友好的结构化数据
"""
import os
import logging
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain.docstore.document import Document
from pydantic import BaseModel, Field

from src.document_loader import DocumentLoader
from src.text_splitter import get_text_splitter
from src.generator import get_generator
from src.utils.helpers import time_function
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, LLM_TEMPERATURE

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 类型定义
class QAPair(BaseModel):
    """问答对模型"""
    question: str = Field(..., description="根据文档内容生成的问题")
    answer: str = Field(..., description="从文档中提取的对应回答")

class StructuredOutput(BaseModel):
    """结构化输出模型"""
    text_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="分块后的文本列表，每块包含内容和元数据")
    qa_pairs: List[QAPair] = Field(default_factory=list, description="从文档内容生成的问答对列表")
    raw_text: Optional[str] = None
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="从文档中提取的表格数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档的元数据")

class DocumentConverter:
    """文档转换器类，将文档转换为结构化数据"""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        llm_provider: str = "openai",
        temperature: float = LLM_TEMPERATURE,
        use_ocr: bool = True,
        api_key: Optional[str] = None
    ):
        """
        初始化文档转换器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            llm_provider: LLM提供商，支持openai和deepseek
            temperature: 生成温度
            use_ocr: 是否使用OCR处理扫描文档
            api_key: API密钥，如果为None则使用环境变量中的密钥
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.use_ocr = use_ocr
        self.api_key = api_key
        
        # 初始化文档加载器
        self.document_loader = DocumentLoader()
        
        # 初始化文本分割器
        self.text_splitter = get_text_splitter(
            splitter_type="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 初始化生成器（用于QA生成）
        try:
            self.generator = get_generator(
                provider=llm_provider,
                temperature=temperature,
                api_key=api_key
            )
            logger.info(f"成功初始化生成器，提供商: {llm_provider}")
        except Exception as e:
            logger.warning(f"初始化生成器失败: {str(e)}，QA生成功能将不可用")
            self.generator = None
        
        logger.info(f"初始化文档转换器: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, "
                   f"llm_provider={llm_provider}, use_ocr={use_ocr}")
    
    @time_function
    def process_document(self, file_path: str, output_formats: List[str] = ["json"]) -> StructuredOutput:
        """
        处理单个文档，转换为结构化数据
        
        Args:
            file_path: 文档路径
            output_formats: 输出格式列表，支持json, jsonl, qa, text_chunks
            
        Returns:
            StructuredOutput对象
        """
        logger.info(f"开始处理文档: {file_path}")
        
        # 加载文档
        if self.use_ocr and file_path.lower().endswith('.pdf'):
            # 对PDF使用OCR处理
            docs = self.document_loader.load_with_ocr(file_path)
        else:
            # 常规加载
            docs = self.document_loader.load_single_document(file_path)
            
        if not docs:
            logger.error(f"无法加载文档: {file_path}")
            return StructuredOutput()
        
        # 提取原始文本
        raw_text = "\n\n".join([doc.page_content for doc in docs])
        
        # 分割文本
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"文档分割完成，共 {len(chunks)} 个文本块")
        
        # 构建结构化输出
        result = StructuredOutput(
            text_chunks=[{
                "content": chunk.page_content,
                "metadata": chunk.metadata
            } for chunk in chunks],
            raw_text=raw_text,
            metadata={
                "source": file_path, 
                "total_chunks": len(chunks),
                "original_document_count": len(docs)
            }
        )
        
        # 生成问答对（如果需要）
        if "qa" in output_formats and self.generator is not None:
            logger.info("开始生成问答对")
            qa_pairs = self._generate_qa_pairs(chunks)
            result.qa_pairs = qa_pairs
            logger.info(f"问答对生成完成，共 {len(qa_pairs)} 对")
        elif "qa" in output_formats and self.generator is None:
            logger.warning("生成器未初始化成功，跳过问答对生成")
            result.qa_pairs = []
        
        # 提取表格（如果有）
        tables = self._extract_tables_from_docs(docs)
        if tables:
            result.tables = tables
            logger.info(f"表格提取完成，共 {len(tables)} 个表格")
        
        logger.info(f"文档处理完成: {file_path}")
        return result
    
    def _generate_qa_pairs(self, chunks: List[Document], qa_per_chunk: int = 3) -> List[QAPair]:
        """
        根据文本块生成问答对
        
        Args:
            chunks: 文本块列表
            qa_per_chunk: 每个块生成的问答对数量
            
        Returns:
            QAPair对象列表
        """
        qa_pairs = []
        
        # 为每个文本块生成问答对
        for i, chunk in enumerate(chunks):
            if i > 5:  # 限制处理的块数，避免生成过多
                break
                
            content = chunk.page_content
            if len(content.strip()) < 100:  # 跳过内容太少的块
                continue
            
            # 构建更明确的提示词
            prompt = f"""
请根据以下文本内容生成{qa_per_chunk}个问答对。请严格按照以下JSON格式回复：

```json
[
    {{{{
        "question": "问题1的内容",
        "answer": "答案1的内容"
    }}}},
    {{{{
        "question": "问题2的内容", 
        "answer": "答案2的内容"
    }}}},
    {{{{
        "question": "问题3的内容",
        "answer": "答案3的内容"  
    }}}}
]
```

要求：
1. 问题必须基于文本内容，答案必须可以从文本中找到
2. 问题要有针对性，测试对文本的理解
3. 答案要准确、简洁
4. 必须生成{qa_per_chunk}个问答对

文本内容：
{content}

请回复JSON格式的问答对：
"""
            
            try:
                # 调用LLM生成问答对
                response = self.generator.llm(prompt)
                logger.debug(f"LLM响应: {response[:200]}...")
                
                # 解析JSON响应
                parsed_qa_pairs = self._parse_qa_response(response)
                
                # 添加到结果列表
                for qa_dict in parsed_qa_pairs:
                    try:
                        qa_pair = QAPair(
                            question=qa_dict["question"].strip(),
                            answer=qa_dict["answer"].strip()
                        )
                        qa_pairs.append(qa_pair)
                        logger.debug(f"成功创建问答对: Q: {qa_pair.question[:50]}...")
                    except Exception as e:
                        logger.warning(f"创建QAPair对象失败: {str(e)}")
                        
            except Exception as e:
                logger.error(f"生成问答对时出错: {str(e)}")
                continue
        
        logger.info(f"总共生成了 {len(qa_pairs)} 个问答对")
        return qa_pairs
    
    def _parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        """
        解析LLM响应中的问答对
        
        Args:
            response: LLM的响应文本
            
        Returns:
            问答对字典列表
        """
        qa_pairs = []
        
        try:
            # 方法1: 尝试解析JSON格式
            import json
            import re
            
            # 提取JSON部分（在```json和```之间，或者在[和]之间）
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 直接查找JSON数组
                json_match = re.search(r'\[.*?\]', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = None
            
            if json_str:
                parsed_data = json.loads(json_str)
                if isinstance(parsed_data, list):
                    for item in parsed_data:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            qa_pairs.append({
                                "question": str(item["question"]).strip(),
                                "answer": str(item["answer"]).strip()
                            })
                    
                    logger.debug(f"JSON解析成功，提取了 {len(qa_pairs)} 个问答对")
                    return qa_pairs
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {str(e)}")
        except Exception as e:
            logger.warning(f"JSON解析出错: {str(e)}")
        
        # 方法2: 尝试正则表达式匹配结构化格式
        try:
            # 匹配各种可能的格式
            patterns = [
                # Q: ... A: ...格式
                r'Q[：:]?\s*(.+?)\s*A[：:]?\s*(.+?)(?=Q[：:]|$)',
                # 问题: ... 答案: ...格式
                r'问题[：:]?\s*(.+?)\s*答案[：:]?\s*(.+?)(?=问题[：:]|$)',
                # 1. 问题... 答案...格式
                r'\d+\.\s*问题[：:]?\s*(.+?)\s*答案[：:]?\s*(.+?)(?=\d+\.|$)',
                # **问题**... **答案**...格式
                r'\*\*问题\*\*[：:]?\s*(.+?)\s*\*\*答案\*\*[：:]?\s*(.+?)(?=\*\*问题\*\*|$)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
                if matches:
                    for question, answer in matches:
                        question = question.strip().replace('\n', ' ')
                        answer = answer.strip().replace('\n', ' ')
                        if len(question) > 5 and len(answer) > 5:
                            qa_pairs.append({
                                "question": question,
                                "answer": answer
                            })
                    
                    if qa_pairs:
                        logger.debug(f"正则表达式解析成功，提取了 {len(qa_pairs)} 个问答对")
                        return qa_pairs
            
        except Exception as e:
            logger.warning(f"正则表达式解析出错: {str(e)}")
        
        # 方法3: 简单的文本分割方法
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            current_question = None
            for line in lines:
                # 跳过格式化字符
                if line in ['```json', '```', '[', ']', '{', '}'] or line.startswith('```'):
                    continue
                
                # 识别问题行
                if any(keyword in line for keyword in ['问题', 'question', 'Q:', '问：', '?', '？']):
                    # 清理问题文本
                    question = re.sub(r'^[^\w\u4e00-\u9fff]*', '', line)
                    question = re.sub(r'问题[：:]?|question[：:]?|Q[：:]?', '', question, flags=re.IGNORECASE).strip()
                    if len(question) > 5:
                        current_question = question
                
                # 识别答案行
                elif current_question and any(keyword in line for keyword in ['答案', 'answer', 'A:', '答：']):
                    # 清理答案文本
                    answer = re.sub(r'^[^\w\u4e00-\u9fff]*', '', line)
                    answer = re.sub(r'答案[：:]?|answer[：:]?|A[：:]?', '', answer, flags=re.IGNORECASE).strip()
                    if len(answer) > 5:
                        qa_pairs.append({
                            "question": current_question,
                            "answer": answer
                        })
                        current_question = None
            
            if qa_pairs:
                logger.debug(f"文本分割解析成功，提取了 {len(qa_pairs)} 个问答对")
                return qa_pairs
                
        except Exception as e:
            logger.warning(f"文本分割解析出错: {str(e)}")
        
        # 如果所有方法都失败，记录原始响应并返回空列表
        logger.error(f"无法解析问答对，原始响应: {response[:500]}...")
        return []
    
    def _extract_tables_from_docs(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        从文档中提取表格数据
        
        Args:
            docs: 文档列表
            
        Returns:
            表格数据列表，每个表格是一个字典
        """
        tables = []
        
        for doc in docs:
            # 检查元数据中是否包含表格数据
            if "table_data" in doc.metadata:
                tables.append({
                    "data": doc.metadata["table_data"],
                    "source_page": doc.metadata.get("page", 1),
                    "sheet_name": doc.metadata.get("sheet_name", "")
                })
        
        return tables
    
    def convert_to_format(self, result: StructuredOutput, output_format: str) -> str:
        """
        将结构化输出转换为指定格式
        
        Args:
            result: 结构化输出对象
            output_format: 输出格式
            
        Returns:
            格式化后的字符串
        """
        if output_format == "json":
            return result.model_dump_json(indent=2)
            
        elif output_format == "jsonl":
            # 将每个文本块转换为一行JSON
            lines = []
            for chunk in result.text_chunks:
                chunk_obj = {
                    "content": chunk["content"],
                    "metadata": chunk["metadata"]
                }
                lines.append(json.dumps(chunk_obj, ensure_ascii=False))
            return "\n".join(lines)
            
        elif output_format == "qa_json":
            # 仅输出问答对的JSON
            qa_list = [qa.model_dump() for qa in result.qa_pairs]
            return json.dumps(qa_list, indent=2, ensure_ascii=False)
            
        elif output_format == "text":
            # 仅输出原始文本
            return result.raw_text or ""
            
        else:
            logger.warning(f"不支持的输出格式: {output_format}")
            return "" 
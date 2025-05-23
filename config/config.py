"""
配置模块，管理应用程序配置参数
"""
import os
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 创建各种目录
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)

EVAL_DATA_DIR = ROOT_DIR / "evaluation_data"
EVAL_DATA_DIR.mkdir(exist_ok=True)

EVAL_RESULTS_DIR = ROOT_DIR / "evaluation_results"
EVAL_RESULTS_DIR.mkdir(exist_ok=True)

# 应用标题和描述
APP_TITLE = "RAG智能问答系统"
APP_DESCRIPTION = "基于检索增强生成技术的智能文档问答系统"

# 嵌入模型配置
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# LLM配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# 默认LLM提供商（openai 或 deepseek）
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "deepseek").lower()

# LLM生成参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# 文本分块配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_CHUNK_SIZE = CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = CHUNK_OVERLAP

# 检索配置
TOP_K = int(os.getenv("TOP_K", "5"))
USE_HYBRID = os.getenv("USE_HYBRID", "True").lower() == "true"
USE_RERANKER = os.getenv("USE_RERANKER", "False").lower() == "true"
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# 评估配置
DEFAULT_EVAL_DATASET = os.getenv("DEFAULT_EVAL_DATASET", "eval_sample.json")

def get_model_config() -> dict:
    """
    获取模型配置
    
    Returns:
        dict: 模型配置字典
    """
    # 首先尝试从配置文件加载
    config_path = ROOT_DIR / "config" / "model_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info(f"从文件加载模型配置: {config_path}")
                return config
        except Exception as e:
            logger.error(f"加载模型配置文件失败: {str(e)}")
    
    # 默认配置
    default_config = {
        # 可用的模型及其提供商
        "models": {
            "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
            "deepseek": ["deepseek-chat", "deepseek-coder"]
        },
        # 默认模型
        "default_model": DEFAULT_LLM_PROVIDER,
        # 回退模型，当默认模型不可用时使用
        "fallback_model": "gpt-3.5-turbo" if DEFAULT_LLM_PROVIDER != "openai" else "deepseek-chat",
        # 模型参数
        "parameters": {
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS
        },
        # 嵌入模型
        "embedding_model": EMBEDDING_MODEL,
        # 向量存储类型
        "vector_store_type": "faiss"
    }
    
    logger.info("使用默认模型配置")
    return default_config

def save_model_config(config: dict) -> bool:
    """
    保存模型配置到文件
    
    Args:
        config: 模型配置字典
        
    Returns:
        bool: 是否成功保存
    """
    config_path = ROOT_DIR / "config" / "model_config.json"
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"已保存模型配置到: {config_path}")
        return True
    except Exception as e:
        logger.error(f"保存模型配置失败: {str(e)}")
        return False 
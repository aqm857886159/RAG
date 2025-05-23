# RAG项目问题排查指南

本文档记录了RAG项目开发和部署过程中遇到的常见问题及其解决方案，特别是针对Windows中文环境的兼容性问题。

## Windows中文环境问题

### 1. FAISS向量存储保存失败

**问题描述**：
在Windows中文环境下，FAISS向量库无法在含有中文或非ASCII字符的路径中正确保存和加载向量数据库。通常会出现以下错误：
```
faiss.RuntimeError: Error al escribir el índice: No such file or directory
```

**原因分析**：
FAISS底层使用C++实现，在处理非ASCII字符路径时存在编码问题，特别是在Windows系统中更为明显。

**解决方案**：
1. 在`vectorizer.py`中实现了路径自动检测和备用路径机制：
   - 检测路径中是否包含非ASCII字符
   - 如包含，则尝试以下备用路径：
     - 用户主目录下的`rag_vector_store`文件夹
     - 系统临时目录下的`rag_vector_store`文件夹
     - 当前工作目录下的`rag_vector_store`文件夹

**相关代码**：
```python
def _get_safe_index_path(self, index_name: str) -> str:
    """获取安全的索引路径，处理中文或特殊字符路径问题"""
    # 原始路径
    original_path = os.path.join(self.vector_store_dir, index_name)
    
    # 检查路径是否包含非ASCII字符
    has_non_ascii = any(ord(c) > 127 for c in original_path)
    
    if not has_non_ascii:
        # 尝试验证路径可用性
        try:
            ensure_directory(original_path)
            # 测试写入权限
            test_file = os.path.join(original_path, ".test_write")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return original_path
        except Exception as e:
            # 继续尝试备用路径
            pass
    
    # 尝试用户主目录、临时目录和当前目录作为备用路径
    # ...
```

### 2. 临时文件处理问题

**问题描述**：
在处理上传的文档时，由于中文文件名可能导致临时文件创建或删除失败。

**解决方案**：
1. 使用Python的`tempfile`模块创建带有随机名称的临时文件
2. 确保在处理完成后正确关闭和删除临时文件
3. 使用`try-finally`块确保临时文件总是被删除

**相关代码**：
```python
def process_document(uploaded_file):
    """处理上传的文档"""
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # 处理文档
        try:
            # 文档处理代码...
        finally:
            # 确保删除临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        return False, f"处理文档时出错: {str(e)}"
```

## 依赖和模型问题

### 1. Pydantic验证器重复注册

**问题描述**：
在使用DeepSeek LLM类时，出现以下错误：
```
ValueError: Validator 'validate_environment' for field 'api_key' has already been registered. Use 'allow_reuse=True' to reuse validators.
```

**原因分析**：
Pydantic 2.x版本中，不允许在继承的模型类中重复注册同名验证器，而我们的LLM类可能存在这种情况。

**解决方案**：
在`@field_validator`装饰器中添加`allow_reuse=True`参数：

```python
@field_validator("api_key", allow_reuse=True)
def validate_environment(cls, field):
    """验证API密钥环境变量"""
    if not field and not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("未设置DeepSeek API密钥")
    return field or os.getenv("DEEPSEEK_API_KEY")
```

### 2. 模型加载错误

**问题描述**：
在加载某些Hugging Face模型时可能出现内存不足或其他错误。

**解决方案**：
1. 针对BGE模型添加特殊配置，控制批处理大小：
```python
if "bge" in embedding_model_name.lower():
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {
        "normalize_embeddings": True,
        "batch_size": 8  # 较小的批处理大小避免OOM
    }
    self.embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
```

2. 添加备用向量存储机制，在FAISS或Chroma失败时使用DocArrayInMemorySearch或SimpleMemoryVectorStore

## 应用启动问题

### 1. 应用启动错误

**问题描述**：
原始的`app.py`在某些环境下存在语法和缩进错误，无法正常启动。

**解决方案**：
1. 创建简化版应用`minimal_app.py`，仅包含核心RAG功能
2. 提供多种启动脚本：
   - `start_minimal.py`: 启动简化版应用
   - `simple_start.py`: 使用修复版app_fixed.py
   - `robust_start.py`: 健壮版启动脚本，自动处理常见问题

### 2. 端口冲突问题

**问题描述**：
启动应用时可能出现端口已被占用的错误：
```
Port 8510 is already in use
```

**解决方案**：
修改启动脚本中的端口号：
```python
# 设置Streamlit命令行参数
port = 8511  # 修改为不同端口
sys.argv = [
    "streamlit", 
    "run", 
    "src/minimal_app.py",
    "--server.port", 
    str(port)
]
```

## 恢复功能指南

随着稳定性问题的解决，可以逐步将高级功能从完整版应用迁移到简化版应用中：

1. **意图识别**：将`intent_recognizer.py`的功能集成到`minimal_app.py`
2. **混合检索**：从`retriever.py`添加混合检索功能
3. **文档结构化转换**：添加专门的页面处理文档转换功能

迁移过程应当遵循以下原则：
1. 每次添加一个功能并充分测试
2. 保持代码简洁和错误处理完善
3. 维护详细的日志记录
4. 针对Windows中文环境进行专门测试 
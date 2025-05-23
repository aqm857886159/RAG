# RAG项目实战 - 快速开始指南

## 🎯 项目概述

**RAG项目实战**是一个基于检索增强生成（RAG）技术的智能文档处理系统，提供双模式功能：
- 🤖 **RAG问答模式**：基于文档内容的智能问答
- 🔄 **文档转换模式**：将文档转换为AI友好的结构化数据

## ⚡ 5分钟快速体验

### 1. 环境准备

**系统要求**：
- Python 3.8+
- 4GB+ 内存
- 网络连接（用于API调用）

**快速安装**：
```bash
# 1. 克隆项目
git clone https://github.com/aqm857886159/RAG.git
cd RAG

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置API密钥
cp .env-example .env
# 编辑 .env 文件，添加你的API密钥
```

### 2. 配置API密钥

编辑 `.env` 文件：

```bash
# OpenAI配置（推荐）
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# 或者 DeepSeek配置（备选）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 默认配置
DEFAULT_LLM_PROVIDER=openai
```

### 3. 启动应用

```bash
# 推荐方式（自动优化配置）
python run_app.py

# 自定义端口
python run_app.py --port 8080

# 备用方式
streamlit run src/app.py
```

### 4. 开始使用

1. 打开浏览器访问：`http://localhost:8501`
2. 选择工作模式（RAG问答 或 文档转换）
3. 上传测试文档
4. 开始体验！

## 🚀 详细安装步骤

### 方式一：标准安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/aqm857886159/RAG.git
cd RAG

# 2. 创建虚拟环境（推荐）
python -m venv venv

# Windows激活
venv\Scripts\activate
# Linux/Mac激活
source venv/bin/activate

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 验证安装
python -c "import streamlit; print('安装成功！')"
```

### 方式二：开发者安装

```bash
# 额外安装开发工具
pip install -r requirements.txt
pip install pytest black flake8 jupyter

# 运行测试验证
python tests/test_core_functions.py
```

## 🔧 配置说明

### 必需配置

**API密钥**：至少配置一个LLM提供商
```bash
# .env 文件示例
OPENAI_API_KEY=sk-xxx...
# 或
DEEPSEEK_API_KEY=sk-xxx...
```

### 可选配置

**高级设置**：
```bash
# 文档处理
CHUNK_SIZE=500              # 文档分块大小
CHUNK_OVERLAP=50            # 块重叠大小
MAX_FILE_SIZE=100           # 最大文件大小(MB)

# 检索配置
TOP_K=5                     # 检索结果数量
USE_HYBRID=true             # 使用混合检索
VECTOR_WEIGHT=0.7           # 向量检索权重

# 生成配置
LLM_TEMPERATURE=0.7         # 生成温度
LLM_MAX_TOKENS=1024         # 最大生成长度
```

## 🎮 使用教程

### RAG问答模式

1. **上传文档**
   ```
   支持格式：PDF、Word、Excel、CSV、TXT
   最大大小：100MB
   批量上传：支持多文件
   ```

2. **配置检索参数**
   - 启用混合检索（推荐）
   - 设置检索数量：3-5个
   - 调整权重比例

3. **开始问答**
   - 输入问题
   - 查看智能回答
   - 点击引用查看来源

**示例问答**：
```
问：这个项目的核心目标是什么？
答：开发一个自动化的解决方案，将常见的文档格式高效地转换为AI应用所需的结构化数据格式。
引用：[文档名称] 第X页
```

### 文档转换模式

1. **选择输入文件**
   - 拖拽上传文件
   - 启用OCR（扫描PDF）
   - 选择处理选项

2. **配置转换参数**
   ```
   分块大小：500字符（推荐）
   重叠度：50字符
   问答对数量：每块3个
   ```

3. **选择输出格式**
   - JSON：结构化数据
   - JSONL：逐行JSON
   - 问答对：纯文本格式
   - 文本块：分块文本

**转换示例**：
```json
{
  "document_id": "doc_001",
  "chunks": [
    {
      "id": "chunk_001",
      "content": "文档内容...",
      "qa_pairs": [
        {
          "question": "什么是RAG？",
          "answer": "检索增强生成..."
        }
      ]
    }
  ]
}
```

## 🧪 测试验证

### 快速测试

```bash
# 核心功能测试
python tests/test_core_functions.py

# 工作功能测试
python tests/test_working_features.py

# 系统完整性测试
python tests/test_complete_system.py
```

### 综合评估

```bash
# 运行完整评估（需要测试文档）
python comprehensive_rag_evaluation.py

# 查看评估报告
cat rag_evaluation_summary.md
```

## 📊 性能优化

### 提升响应速度

1. **使用GPU加速**（如果有GPU）
2. **调整分块大小**：较小分块=更快检索
3. **减少检索数量**：TOP_K=3
4. **启用缓存**：重复查询更快

### 提升准确率

1. **使用混合检索**：向量+BM25
2. **启用重排序**：更精确的结果
3. **调整检索权重**：根据文档类型优化
4. **增加上下文**：TOP_K=5-7

## 🔍 常见问题

### 安装问题

**Q: pip安装失败**
```bash
# 解决方案
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

**Q: FAISS安装问题**
```bash
# CPU版本
pip install faiss-cpu

# GPU版本（如果有CUDA）
pip install faiss-gpu
```

### 运行问题

**Q: API调用失败**
- 检查网络连接
- 验证API密钥
- 检查余额/配额

**Q: 内存不足**
- 减小CHUNK_SIZE
- 降低TOP_K值
- 分批处理文档

**Q: 中文路径问题**
- 系统自动处理
- 使用英文路径更稳定

### 功能问题

**Q: OCR识别不准确**
- 确保图片清晰
- 调整OCR参数
- 预处理图像质量

**Q: 检索结果不相关**
- 检查文档质量
- 调整检索参数
- 使用混合检索

## 📈 进阶使用

### 自定义配置

1. **创建配置文件**：`config/custom_settings.py`
2. **自定义提示词**：修改生成模板
3. **添加新格式**：扩展文档加载器
4. **集成新模型**：添加LLM提供商

### API集成

```python
# 示例：程序化调用
from src.retriever import get_retriever
from src.generator import get_generator

# 初始化系统
retriever = get_retriever()
generator = get_generator()

# 问答
result = generator.generate("你的问题")
print(result['answer'])
```

### 批量处理

```python
# 批量文档转换
from src.data_converter import DocumentConverter

converter = DocumentConverter()
results = converter.batch_convert(file_list)
```

## 📞 获取帮助

1. **查看文档**：`docs/` 目录
2. **运行测试**：确认系统状态
3. **检查日志**：`logs/` 目录
4. **GitHub Issues**：报告问题

## 🎉 下一步

1. **探索高级功能**：混合检索、重排序
2. **自定义配置**：根据需求调整参数
3. **集成到项目**：使用API接口
4. **贡献代码**：提交改进建议

---

**恭喜！你已经成功启动了RAG项目实战系统！** 🎉

现在可以开始探索智能文档问答和转换功能了。

*如有问题，请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解项目结构* 
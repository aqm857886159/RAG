# RAG项目实战 - 智能文档问答与转换系统

## 项目简介

本项目是一个基于RAG（检索增强生成）技术的**双模式智能系统**，集成了：

1. **智能文档问答系统**：基于RAG技术的知识库问答，支持多文档格式，提供准确且可追溯的回答
2. **文档结构化转换系统**：将多格式文档转换为AI友好的结构化数据，支持OCR处理和问答对生成

该系统特别适合企业知识库问答、专业领域技术咨询、文档智能分析，以及需要可追溯性的决策支持场景。

## 🚀 最新更新

### V3.0 版本 - 项目结构优化

1. **项目架构重新整理**：
   - 清理冗余启动文件，统一启动入口
   - 测试文件统一移至 `tests/` 目录
   - 工具脚本移至 `scripts/` 目录
   - 创建 `examples/` 目录用于示例代码
   - 备份历史文件至 `backup/` 目录

2. **双模式集成完成**：
   - RAG问答系统与文档转换系统完美融合
   - 统一的Streamlit界面，模式切换简单直观
   - 共享核心组件，提高系统一致性

3. **增强的文档处理能力**：
   - 支持Excel文件处理（.xlsx, .xls）
   - OCR处理能力（PaddleOCR + PyMuPDF）
   - 智能表格提取和结构化输出
   - 多种输出格式（JSON、JSONL、问答对、文本块）

4. **完善的测试框架**：
   - 独立的测试目录结构
   - 转换功能测试覆盖
   - 兼容性测试确保系统稳定性

## 🔧 核心功能

### 智能问答系统
1. **多格式文档处理**：支持PDF、Word、Excel、CSV、TXT等格式
2. **混合检索技术**：向量检索 + BM25关键词检索 + 语义重排序
3. **增强生成**：基于检索结果生成高质量回答，带引用标记
4. **意图识别**：自动识别用户问题类型，优化处理流程
5. **对话历史管理**：支持多轮对话，智能上下文理解

### 文档转换系统
1. **多格式输入**：PDF（含扫描件）、Word、Excel、CSV、TXT
2. **OCR处理**：自动识别扫描文档中的文本内容
3. **智能分块**：可配置的文本分块策略
4. **问答对生成**：基于LLM自动生成高质量问答对
5. **多格式输出**：JSON、JSONL、问答对文本、纯文本块
6. **表格数据提取**：自动提取并保留表格结构信息

## 📁 项目结构

```
RAG项目实战/
├── 📁 src/                     # 源代码目录
│   ├── app.py                 # 主应用程序（双模式界面）
│   ├── document_loader.py     # 文档加载模块（支持Excel+OCR）
│   ├── data_converter.py      # 文档转换模块
│   ├── vectorizer.py          # 向量化模块
│   ├── retriever.py           # 检索模块（混合检索）
│   ├── fixed_generator.py     # 生成模块（修复版）
│   ├── intent_recognizer.py   # 意图识别模块
│   ├── evaluator.py           # 评估模块
│   └── components/            # UI组件
├── 📁 tests/                   # 测试目录
│   ├── test_converter.py      # 转换功能测试
│   ├── test_rag.py            # RAG功能测试
│   ├── test_compatibility.py  # 兼容性测试
│   └── simple_test.py         # 简单测试脚本
├── 📁 scripts/                 # 工具脚本
│   ├── organize_project.py    # 项目整理脚本
│   ├── clean_project.py       # 项目清理脚本
│   └── start.py               # 自定义启动脚本
├── 📁 data/                    # 原始数据文件
├── 📁 output/                  # 转换结果输出
├── 📁 vector_store/           # 向量数据库存储
├── 📁 examples/               # 示例代码和数据
├── 📁 docs/                   # 项目文档
├── 📁 backup/                 # 历史文件备份
├── 📁 config/                 # 配置文件
├── 📁 utils/                  # 工具函数
├── 📄 run_app.py              # 🎯 推荐启动脚本
├── 📄 app.py                  # 备用启动入口
├── 📄 requirements.txt        # 项目依赖
└── 📄 README.md               # 项目说明
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 复制配置文件
copy .env-example .env

# 编辑 .env 文件，配置你的API密钥
```

### 2. 启动应用

#### 🎯 推荐方式（自动选择最佳配置）：
```bash
python run_app.py
```

#### 自定义启动选项：
```bash
# 指定端口和主机
python run_app.py --port 8080 --host 0.0.0.0

# 指定应用版本
python run_app.py --app src/app.py
```

#### 备用启动方式：
```bash
python app.py
```

### 3. 使用系统

1. **RAG问答模式**：
   - 上传文档到知识库
   - 配置检索参数（混合检索、重排序等）
   - 开始智能问答

2. **文档转换模式**：
   - 上传需要转换的文档
   - 选择输出格式和处理选项
   - 获取结构化数据结果

## 🔧 详细使用指南

### RAG问答系统配置

1. **上传文档**：
   - 支持PDF、Word、Excel、CSV、TXT格式
   - 批量上传多个文档
   - 自动文档预处理和分块

2. **检索配置**：
   - **混合检索**：启用向量检索 + BM25检索
   - **语义重排序**：使用Cross-Encoder提高精度
   - **检索数量**：设置返回结果数量（推荐3-5）
   - **权重调节**：调整向量检索和BM25的权重比例

3. **生成配置**：
   - **模型选择**：OpenAI GPT / DeepSeek
   - **生成参数**：温度、最大长度等
   - **引用模式**：自动标注引用来源

### 文档转换系统配置

1. **输入配置**：
   - **文件上传**：拖拽或点击上传
   - **OCR选项**：对扫描PDF启用OCR处理
   - **格式识别**：自动识别文档格式

2. **处理配置**：
   - **分块参数**：块大小（默认500）、重叠度（默认50）
   - **问答生成**：每块生成的问答对数量（默认3）
   - **表格处理**：是否提取表格数据

3. **输出配置**：
   - **JSON格式**：结构化JSON输出
   - **JSONL格式**：逐行JSON格式
   - **问答对文本**：纯文本问答对
   - **文本块**：分块后的纯文本

4. **高级选项**：
   - **并发处理**：OCR线程数设置
   - **质量控制**：问答对质量过滤
   - **批量处理**：多文档批量转换

## 📊 系统评估

### 评估指标
- **检索评估**：精确率、召回率、F1分数
- **生成评估**：流畅度、相关性、事实准确性
- **系统评估**：响应时间、资源使用、稳定性

### 运行评估
```bash
# 运行完整评估
cd evaluation/scripts
python run_evaluation.py

# 运行特定评估
python evaluate_conversation.py
```

## ⚙️ 配置参数

### 环境变量配置（.env文件）
```bash
# OpenAI配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# DeepSeek配置  
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 默认配置
DEFAULT_LLM_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=BAAI/bge-large-zh
```

### 应用参数配置
```python
# 文档处理参数
CHUNK_SIZE = 500          # 文档分块大小
CHUNK_OVERLAP = 50        # 块重叠大小
MAX_FILE_SIZE = 100       # 最大文件大小(MB)

# 检索参数
TOP_K = 5                 # 检索结果数量
USE_HYBRID = True         # 使用混合检索
VECTOR_WEIGHT = 0.7       # 向量检索权重
BM25_WEIGHT = 0.3         # BM25检索权重

# 生成参数
LLM_TEMPERATURE = 0.7     # 生成温度
LLM_MAX_TOKENS = 1024     # 最大生成长度

# 转换参数
USE_OCR = True            # 使用OCR处理
QA_PER_CHUNK = 3          # 每块问答对数量
OCR_THREADS = 4           # OCR处理线程数
```

## 🔍 测试系统

### 运行测试
```bash
# 转换功能测试
cd tests
python test_converter.py

# RAG功能测试
python test_rag.py

# 兼容性测试
python test_compatibility.py

# 简单功能测试
python simple_test.py
```

### 测试覆盖
- ✅ 文档加载和处理
- ✅ 向量化和检索
- ✅ 文档转换功能
- ✅ OCR处理能力
- ✅ 问答生成质量
- ✅ 系统兼容性

## 🐛 常见问题与解决方案

### 1. Windows中文路径问题
**问题**：FAISS向量库无法在中文路径下正常工作
**解决方案**：系统自动检测并切换到兼容路径（用户目录→临时目录→当前目录）

### 2. OCR处理失败
**问题**：扫描PDF的OCR处理失败或速度慢
**解决方案**：
- 确保安装PaddleOCR依赖：`pip install paddlepaddle paddleocr`
- 调整OCR线程数：在界面中设置较少的线程数
- 对于不需要OCR的文档，关闭OCR选项

### 3. 内存不足错误
**问题**：处理大文档时出现内存不足
**解决方案**：
- 减小分块大小（CHUNK_SIZE）
- 降低检索结果数量（TOP_K）
- 分批处理大文档

### 4. API调用失败
**问题**：LLM API调用失败或超时
**解决方案**：
- 检查.env文件中的API密钥配置
- 确认网络连接和API服务状态
- 使用备用模型提供商

### 5. 依赖安装问题
**问题**：某些依赖包安装失败
**解决方案**：
```bash
# 逐步安装问题依赖
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-cache-dir
```

## 🛠️ 开发指南

### 添加新功能
1. 在 `src/` 目录下创建新模块
2. 在 `tests/` 目录下添加对应测试
3. 更新 `requirements.txt` 依赖
4. 更新 README 文档

### 代码规范
- 遵循PEP 8 Python代码风格
- 使用类型提示（Type Hints）
- 编写详细的文档字符串
- 添加适当的错误处理

## 📈 性能优化建议

1. **向量数据库优化**：
   - 使用适当的向量维度
   - 定期清理无用的向量数据
   - 考虑使用分布式向量数据库

2. **检索优化**：
   - 调整混合检索权重
   - 使用更精确的重排序模型
   - 优化检索结果数量

3. **生成优化**：
   - 选择合适的模型大小
   - 调整生成参数
   - 使用模型缓存机制

## 🔮 未来计划

### 短期计划（1-2个月）
- [ ] 支持更多文档格式（PPT、Markdown等）
- [ ] 添加文档结构化预览功能
- [ ] 优化OCR处理速度和准确率
- [ ] 增加批量文档处理API

### 中期计划（3-6个月）
- [ ] 多模态RAG（图像、音频内容）
- [ ] 实时知识库更新机制
- [ ] 分布式部署支持
- [ ] 企业级权限管理

### 长期计划（6个月以上）
- [ ] Agent架构集成
- [ ] 自动化文档分析报告
- [ ] 知识图谱构建
- [ ] 多语言支持

## 📞 支持与反馈

如果你在使用过程中遇到问题或有改进建议：

1. 查看常见问题解决方案
2. 检查 `logs/` 目录中的日志文件
3. 运行 `tests/` 目录中的相关测试
4. 提供详细的错误信息和使用环境

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

---

**感谢你使用RAG项目实战系统！🎉**

通过整合RAG问答和文档转换功能，本系统为你提供了一个完整的文档智能处理解决方案。无论是企业知识库问答还是文档结构化转换，都能轻松应对。

推荐启动命令：`python run_app.py` 
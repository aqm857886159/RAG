# RAG项目实战 - 项目结构说明

## 📁 项目目录结构

```
RAG项目实战/
├── 📁 src/                          # 核心源代码目录
│   ├── 🎯 app.py                   # 主Streamlit应用（推荐）
│   ├── 📄 app_fixed.py             # 修复版应用
│   ├── 🚀 minimal_app.py           # 最小版本应用
│   ├── 📚 document_loader.py       # 文档加载模块
│   ├── 🔄 data_converter.py        # 文档转换模块
│   ├── 🧠 intent_recognizer.py     # 意图识别模块
│   ├── 🤖 generator.py             # 回答生成模块
│   ├── 🗄️ vectorizer.py            # 向量化模块
│   ├── 🔍 retriever.py             # 检索模块
│   ├── ⚡ llm.py                   # LLM接口模块
│   ├── ✂️ text_splitter.py         # 文本分块模块
│   ├── 📊 evaluator.py             # 评估模块
│   ├── 🔧 chunker.py               # 分块处理器
│   ├── 🔀 converter.py             # 转换器
│   ├── 📦 custom_loaders.py        # 自定义加载器
│   ├── 🎛️ custom_retrievers.py     # 自定义检索器
│   ├── 📁 components/              # UI组件
│   ├── 📁 utils/                   # 工具函数
│   └── 📁 models/                  # 模型相关
│
├── 📁 tests/                        # 测试目录
│   ├── 🧪 test_core_functions.py   # 核心功能测试
│   ├── 🔬 test_working_features.py # 工作功能测试
│   ├── 🌍 test_complete_system.py  # 完整系统测试
│   └── 🔧 test_fixed_functions.py  # 修复功能测试
│
├── 📁 docs/                         # 项目文档
│   ├── 📋 PROJECT_STRUCTURE.md     # 项目结构说明（本文件）
│   ├── 🚀 QUICK_START.md           # 快速开始指南
│   ├── 🔧 API_DOCUMENTATION.md     # API文档
│   ├── 📊 EVALUATION_GUIDE.md       # 评估指南
│   ├── 📊 PROJECT_SUMMARY.md       # 项目总结报告
│   └── 📋 RELEASE_NOTES.md         # 发布说明
│
├── 📁 scripts/                      # 工具脚本
│   ├── 🏃 run_app.py               # 应用启动脚本
│   ├── 🔧 run_fixed_app.py         # 修复版启动脚本
│   └── 📱 app.py                   # 备用启动脚本
│
├── 📁 evaluation/                   # 评估系统和结果
│   ├── 📊 comprehensive_rag_evaluation.py  # 全面RAG评估脚本
│   ├── 📈 evaluation_results/      # 评估结果
│   └── 📋 evaluation_data/         # 评估数据
│   └── scripts/                    # 评估相关脚本
├── 📁 data/                         # 数据目录
│   ├── 📄 input/                   # 输入文档
│   └── 📊 samples/                 # 示例数据
│
├── 📁 output/                       # 输出目录
│   ├── 📊 converted/               # 转换结果
│   ├── 📋 reports/                 # 评估报告
│   └── 📄 exports/                 # 导出文件
│
├── 📁 config/                       # 配置文件
│   ├── ⚙️ settings.py              # 应用设置
│   └── 🔑 .env-example             # 环境变量示例
│
├── 📁 vector_store/                 # 向量存储
│   ├── 🗄️ faiss_indices/           # FAISS索引
│   └── 📚 embeddings/              # 嵌入向量
│
├── 📁 models/                       # 模型文件
│   ├── 🤖 llm_models/              # 语言模型
│   └── 🧮 embedding_models/        # 嵌入模型
│
├── 📁 logs/                         # 日志文件
│   ├── 📝 app.log                  # 应用日志
│   └── 🔍 debug.log                # 调试日志
│
├── 📁 backup/                       # 备份文件
│   └── 🗂️ archived/                # 历史文件
│
├── 📁 examples/                     # 示例代码
│   ├── 📚 sample_documents/        # 示例文档
│   └── 💻 demo_notebooks/          # 演示笔记本
│
├── 📁 utils/                        # 通用工具
│   ├── 🔧 helpers.py               # 辅助函数
│   └── 🛠️ common.py                # 通用工具
│
├── 📁 tmp/                          # 临时文件
│   └── 🗑️ cache/                   # 缓存文件
│
├── 📄 README.md                     # 项目主文档
├── 📊 PROJECT_SUMMARY.md            # 项目总结报告
├── 📈 rag_evaluation_summary.md     # RAG评估总结
├── 📋 ARCHITECTURE_IMPLEMENTATION_STATUS.md  # 架构实现状态
├── 🎯 FINAL_ARCHITECTURE_STATUS.md  # 最终架构状态
├── 📊 PROJECT_STATUS.md             # 项目状态
├── 📦 requirements.txt              # 依赖包列表
├── ⚙️ .env-example                  # 环境变量示例
├── 🚫 .gitignore                   # Git忽略文件
└── 🔍 .cursorrules                 # Cursor编辑器规则
```

## 🎯 核心模块说明

### 📱 应用入口
- **src/app.py**: 主Streamlit应用，双模式界面（RAG问答+文档转换）
- **run_app.py**: 推荐的启动脚本，自动选择最佳配置

### 🧠 核心组件
- **intent_recognizer.py**: 智能意图识别，支持多种查询类型
- **document_loader.py**: 多格式文档加载，支持PDF、Word、Excel等
- **vectorizer.py**: 向量化处理，FAISS+BGE嵌入模型
- **retriever.py**: 混合检索系统，向量+BM25+重排序
- **generator.py**: 智能回答生成，支持RAG和纯LLM模式
- **data_converter.py**: 文档结构化转换，多格式输出

### 🔧 支持模块
- **llm.py**: LLM接口封装，支持OpenAI、DeepSeek等
- **text_splitter.py**: 智能文本分块，递归字符策略
- **evaluator.py**: 系统性能评估，多维度指标

### 🧪 测试框架
- **test_core_functions.py**: 核心功能单元测试
- **test_working_features.py**: 工作功能集成测试
- **comprehensive_rag_evaluation.py**: 全面系统评估

## 📊 文档体系

### 🎯 用户文档
- **README.md**: 项目总览、快速开始、功能介绍
- **docs/QUICK_START.md**: 详细的快速开始指南
- **docs/API_DOCUMENTATION.md**: API接口文档

### 📈 技术文档
- **PROJECT_SUMMARY.md**: 完整的项目技术总结
- **rag_evaluation_summary.md**: RAG系统评估报告
- **ARCHITECTURE_IMPLEMENTATION_STATUS.md**: 架构实现详情

### 🔍 评估报告
- **evaluation_results/**: 性能评估结果数据
- **rag_evaluation_report_*.json**: 详细的评估数据

## 🚀 启动方式

### 推荐启动（自动优化）
```bash
python run_app.py
```

### 自定义启动
```bash
python run_app.py --port 8080 --host 0.0.0.0
```

### 直接启动
```bash
streamlit run src/app.py
```

## 📦 依赖管理

- **requirements.txt**: 生产环境依赖
- **.env-example**: 环境变量配置模板
- **config/**: 应用配置文件

## 🗄️ 数据流向

```
data/ (输入) → src/ (处理) → output/ (输出)
                ↓
          vector_store/ (存储)
                ↓
            logs/ (日志)
```

## 🔧 开发建议

1. **新功能开发**: 在src/目录下创建新模块
2. **测试编写**: 在tests/目录下添加对应测试
3. **文档更新**: 在docs/目录下更新相关文档
4. **配置修改**: 在config/目录下管理配置文件

## 📋 维护说明

- **backup/**: 包含历史版本和重要文件备份
- **logs/**: 应用运行日志，便于问题排查
- **tmp/**: 临时文件和缓存，可定期清理
- **vector_store/**: 向量数据库，占用空间较大

---

*最后更新: 2025-05-23 | 版本: v2.0* 
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多轮对话评估数据集生成脚本
用于评估指代补全和文档引用功能
"""
import os
import json
import logging
from pathlib import Path

# 添加项目根目录到Python路径
import sys
from pathlib import Path

# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
# 获取项目根目录（evaluation/scripts的上两级目录）
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# 将项目根目录添加到Python路径
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 定义目录
EVAL_DIR = SCRIPT_DIR.parent
DATA_DIR = EVAL_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# 预设多轮对话数据
CONVERSATION_DATA = [
    {
        "id": "Conv_001",
        "title": "深度学习模型比较",
        "conversation": [
            {"role": "user", "content": "BERT和GPT有什么区别？"},
            {"role": "assistant", "content": "BERT(Bidirectional Encoder Representations from Transformers)和GPT(Generative Pre-trained Transformer)的主要区别在于：\n\n1. 架构设计：BERT是双向的Transformer编码器模型，而GPT是单向的Transformer解码器模型\n2. 预训练任务：BERT使用掩码语言模型(MLM)和下一句预测(NSP)任务，GPT使用自回归语言模型\n3. 应用场景：BERT适合理解任务(如分类、问答)，GPT适合生成任务(如文本生成、聊天机器人)\n4. 语境理解：BERT可以同时利用上下文信息，GPT只能利用左侧上下文"},
            {"role": "user", "content": "它们在训练方式上有什么不同？"}
        ],
        "reference_answer": "BERT和GPT在训练方式上的主要区别：\n1. BERT使用掩码语言模型(MLM)，随机遮盖输入中15%的词并预测它们，允许模型同时利用左右上下文；而GPT采用自回归训练方式，只预测下一个词，每个词只能看到它前面的词。\n2. BERT使用了句子对训练和下一句预测(NSP)任务帮助学习句子关系；GPT没有专门建模句子关系。\n3. BERT可以同时从双向上下文获取信息进行预测；GPT只能从左向右单向获取信息。",
        "reference_documents": ["doc_bert_gpt_comparison"],
        "expected_keywords": ["掩码语言模型", "自回归", "双向", "MLM", "NSP"],
        "resolved_question": "BERT和GPT在训练方式上有什么不同？"
    },
    {
        "id": "Conv_002",
        "title": "中国GDP增长",
        "conversation": [
            {"role": "user", "content": "中国2022年GDP增长情况如何？"},
            {"role": "assistant", "content": "根据国家统计局数据，中国2022年GDP增长了3.0%，达到121.02万亿元人民币[1]。这一增速虽然低于前一年，但在全球主要经济体中表现相对稳健，尤其考虑到新冠疫情的影响[2]。\n\n[1] 国家统计局《2022年国民经济和社会发展统计公报》\n[2] 中国人民银行《2022年第四季度中国货币政策执行报告》"},
            {"role": "user", "content": "那2023年的情况呢？"}
        ],
        "reference_answer": "根据国家统计局发布的数据，中国2023年GDP同比增长5.2%，达到126.06万亿元人民币。这一增长率超过了政府此前设定的"约5.0%"的目标，显示出经济复苏的韧性和活力。分季度看，第一季度同比增长4.5%，第二季度增长6.3%，第三季度增长4.9%，第四季度增长5.2%。从产业结构看，第一产业增加值8.79万亿元，增长4.1%；第二产业增加值49.16万亿元，增长4.7%；第三产业增加值68.11万亿元，增长5.8%。",
        "reference_documents": ["doc_china_gdp_2023"],
        "expected_keywords": ["5.2%", "126.06万亿", "季度", "产业结构"],
        "resolved_question": "中国2023年GDP增长情况如何？"
    },
    {
        "id": "Conv_003",
        "title": "Python和Java比较",
        "conversation": [
            {"role": "user", "content": "Python和Java在性能方面有什么区别？"},
            {"role": "assistant", "content": "Python和Java在性能上的主要区别：\n\n1. 执行方式：Java代码先编译为字节码，再由JVM执行，而Python是解释型语言，一边解释一边执行，所以Java通常比Python运行更快[1]。\n2. 内存管理：两者都有自动垃圾回收，但Java的内存管理更高效，尤其在大型应用中表现更好[2]。\n3. 并发处理：Java的多线程支持更成熟，适合处理高并发任务；Python因为GIL(全局解释器锁)的存在，多线程性能受限[1]。\n4. 适用场景：Java更适合计算密集型应用、企业级应用和Android开发；Python适合数据分析、AI和快速开发原型[2]。\n\n[1] 《Programming Language Performance Comparison》\n[2] 《Java vs Python: A Comprehensive Analysis》"},
            {"role": "user", "content": "在web开发方面哪个更好？"}
        ],
        "reference_answer": "在Web开发方面，Python和Java各有优势：\n\nPython优势：\n1. 开发速度快：使用Django或Flask等框架，可以快速构建web应用\n2. 代码简洁：更少的代码量完成相同功能\n3. 灵活性高：适合原型开发和中小型应用\n4. 丰富的库：如Django、Flask、Pyramid等成熟Web框架\n5. 适合数据密集型应用：与数据分析工具无缝集成\n\nJava优势：\n1. 性能更好：处理高并发负载能力强\n2. 企业级支持：Spring、JavaEE等成熟框架适合大型企业应用\n3. 安全性高：类型安全和成熟的安全机制\n4. 可扩展性好：适合构建大型、复杂的系统\n5. 移动开发整合：可与Android开发生态系统整合\n\n选择取决于项目需求：小型快速开发项目选Python，大型企业级应用选Java更合适。",
        "reference_documents": ["doc_python_java_web_dev"],
        "expected_keywords": ["Django", "Flask", "Spring", "性能", "开发速度"],
        "resolved_question": "在web开发方面Python和Java哪个更好？"
    },
    {
        "id": "Conv_004",
        "title": "气候变化影响",
        "conversation": [
            {"role": "user", "content": "气候变化对农业生产有什么影响？"},
            {"role": "assistant", "content": "气候变化对农业生产的主要影响包括：\n\n1. 温度升高：改变作物生长季，某些地区可能延长生长期，但过高温度会导致作物热应激，降低产量[1]。\n2. 降水模式变化：导致干旱或洪涝增加，影响灌溉和土壤质量[1]。\n3. 病虫害增加：气候变暖使某些病虫害扩大分布范围，增加爆发频率[2]。\n4. 授粉减少：对传粉昆虫的生活周期产生影响，可能降低授粉效率[2]。\n5. 种植区域变化：某些作物适宜种植区域向高纬度或高海拔地区移动[1]。\n6. 极端天气事件：强风暴、热浪等极端天气更频繁，直接毁坏农作物[3]。\n7. 海平面上升：导致沿海农业区盐碱化，减少可耕地面积[3]。\n\n[1] 联合国粮农组织《气候变化与粮食安全》报告\n[2] 《气候变化对全球农业生态系统的影响》研究\n[3] IPCC《气候变化与土地特别报告》"},
            {"role": "user", "content": "中国针对这些问题有什么应对措施？"}
        ],
        "reference_answer": "中国针对气候变化对农业影响的主要应对措施包括：\n\n1. 农作物品种改良：培育耐热、耐旱、抗病虫害的新品种，如"超级稻"和抗旱小麦\n2. 调整种植结构：根据气候变化趋势，科学调整农作物种植区域和种植季节\n3. 发展节水农业：推广滴灌、喷灌等高效节水灌溉技术，建设农田水利设施\n4. 加强农业保险：扩大农业保险覆盖面，提高对极端天气和自然灾害的抵御能力\n5. 发展智慧农业：利用大数据、物联网等现代信息技术，实现精准农业管理\n6. 加强农业碳汇：推广秸秆还田、保护性耕作等技术，增加土壤碳储量\n7. 建设气候适应型农业：统筹考虑气候变化因素，构建适应气候变化的农业生产体系\n8. 强化监测预警：建立农业气象灾害监测预警系统，提高预警精度和时效性\n\n这些措施已纳入国家"十四五"规划和2035年远景目标，以提高中国农业对气候变化的适应能力和粮食安全保障水平。",
        "reference_documents": ["doc_china_climate_measures", "doc_china_agriculture_policy"],
        "expected_keywords": ["品种改良", "节水农业", "智慧农业", "农业保险", "监测预警"],
        "resolved_question": "中国针对气候变化对农业的影响有什么应对措施？"
    },
    {
        "id": "Conv_005",
        "title": "区块链技术",
        "conversation": [
            {"role": "user", "content": "区块链技术的核心原理是什么？"},
            {"role": "assistant", "content": "区块链技术的核心原理包括以下几个方面：\n\n1. 分布式账本：所有参与节点共同维护一个分布式数据库，每个节点都拥有完整账本副本[1]。\n2. 共识机制：通过算法（如工作量证明PoW、权益证明PoS等）确保所有节点对交易达成一致，解决拜占庭将军问题[1]。\n3. 密码学基础：使用公钥密码学和哈希函数保证数据安全性和完整性[2]。\n4. 区块结构：交易数据被打包成区块，每个区块包含前一区块的哈希值，形成链式结构[2]。\n5. 不可篡改性：一旦数据被确认并加入区块链，除非控制超过51%的算力，否则几乎不可能篡改[3]。\n6. 去中心化：没有中央管理机构，依靠网络中的节点共同维护系统运行[3]。\n\n[1] 《区块链技术与应用》，中国科学院信息技术研究所\n[2] Nakamoto, S.《Bitcoin: A Peer-to-Peer Electronic Cash System》\n[3] 《区块链：定义未来金融与经济新格局》，长江商学院"},
            {"role": "user", "content": "它在金融行业有哪些应用？"}
        ],
        "reference_answer": "区块链技术在金融行业的主要应用包括：\n\n1. 跨境支付：利用区块链技术降低跨境转账成本，提高交易速度，如Ripple网络已被多家银行采用\n2. 数字货币：包括中央银行数字货币(CBDC)和稳定币，提供更高效安全的支付方式\n3. 证券交易：通过区块链技术实现证券交易的清算和结算过程自动化，缩短交易周期，如澳大利亚证券交易所(ASX)的区块链清算系统\n4. 智能合约：自动执行金融合约条款，用于保险理赔、衍生品交易、贷款发放等，减少中介环节\n5. 供应链金融：提高供应链金融透明度和效率，降低融资成本，如IBM和马士基的TradeLens平台\n6. 身份验证：提供安全可靠的数字身份验证系统，简化KYC(了解你的客户)和AML(反洗钱)流程\n7. 资产通证化：将实物资产(如房地产、艺术品)转化为区块链上的数字通证，提高流动性\n8. 去中心化金融(DeFi)：构建不依赖传统金融中介的借贷、交易和资产管理服务\n\n这些应用正在重塑金融行业的业务流程和商业模式，提高效率的同时降低运营成本和风险。",
        "reference_documents": ["doc_blockchain_finance", "doc_defi_applications"],
        "expected_keywords": ["跨境支付", "数字货币", "智能合约", "去中心化金融", "资产通证化"],
        "resolved_question": "区块链技术在金融行业有哪些应用？"
    }
]

def create_conversation_dataset(output_file=None):
    """创建多轮对话评估数据集"""
    if output_file is None:
        output_file = DATA_DIR / "conversation_eval_dataset.json"
    
    logger.info(f"创建多轮对话评估数据集: {output_file}")
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(CONVERSATION_DATA, f, ensure_ascii=False, indent=2)
    
    logger.info(f"多轮对话评估数据集已生成: {output_file}")
    logger.info(f"共{len(CONVERSATION_DATA)}个对话样例")
    
    return output_file

def main():
    """主函数"""
    create_conversation_dataset()
    
    # 打印示例
    print("\n对话样例:")
    print(f"ID: {CONVERSATION_DATA[0]['id']}")
    print(f"标题: {CONVERSATION_DATA[0]['title']}")
    for i, msg in enumerate(CONVERSATION_DATA[0]['conversation']):
        role = "用户" if msg['role'] == 'user' else "助手"
        print(f"{role}: {msg['content'][:50]}..." if len(msg['content']) > 50 else f"{role}: {msg['content']}")
    print(f"补全后的问题: {CONVERSATION_DATA[0]['resolved_question']}")
    print("---")
    print("此数据集可用于评估指代补全和文档引用功能")

if __name__ == "__main__":
    main() 
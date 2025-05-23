#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试DeepSeek API配置是否正确
"""
import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 加载环境变量
load_dotenv()

# 打印API配置
print("===== DeepSeek API配置 =====")
print(f"API密钥: {os.getenv('DEEPSEEK_API_KEY')}")
print(f"API基础URL: {os.getenv('DEEPSEEK_API_BASE')}")
print(f"使用的模型: {os.getenv('DEEPSEEK_MODEL')}")
print(f"默认LLM提供商: {os.getenv('DEFAULT_LLM_PROVIDER')}")

# 尝试向DeepSeek API发送一个简单请求
try:
    # 使用项目自定义的DeepSeek LLM类
    from src.models.deepseek_llm import DeepSeekLLM
    
    print("\n===== 测试API连接 =====")
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    deepseek_api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    
    # 创建DeepSeekLLM实例
    llm = DeepSeekLLM(
        api_key=deepseek_api_key,
        model_name=deepseek_model,
        api_base=deepseek_api_base
    )
    
    # 发送测试请求
    response = llm.invoke("你好，请用中文回答：今天是星期几？")
    print(f"API响应: {response}")
    print("\n✅ API连接测试成功!")
except ImportError as ie:
    print(f"\n❌ 导入错误: {str(ie)}")
    print("请确保已安装必要的库")
except Exception as e:
    print(f"\n❌ API连接测试失败: {str(e)}")

print("\n配置测试完成!") 
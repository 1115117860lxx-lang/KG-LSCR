# -*- coding: utf-8 -*-
"""
KG-LSCR Benchmark 核心配置文件
定义模型基座、推理参数、数据路径与评测指标配置
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 数据集配置
DATA_CONFIG = {
    "law_library_path": os.path.join(DATA_DIR, "law_library.json"),
    "sample_cases_path": os.path.join(DATA_DIR, "sample_cases.json"),
    "gold_cases_path": os.path.join(DATA_DIR, "gold100_sample.json"),
    "full_test_2000_path": os.path.join(DATA_DIR, "cail2018_traffic_test_2000.json"),
}

# 统一评测模型配置
# 支持 "reproduction" (极速重现/Mock模式无需GPU), "vllm" (本地高吞吐), "transformers", "api" (OpenAI/DeepSeek API)
MODEL_CONFIG = {
    "engine": "reproduction",  # 默认在无GPU环境提供确定性高保真复现输出
    "base_model_name": "Qwen/Qwen2.5-72B-Instruct",
    "alternative_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-70B",
    "temperature": 0.1,        # 受控低温解码，杜绝随机性
    "top_p": 0.9,
    "max_new_tokens": 1536,
    "api_base": os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
}

# 检索配置 (DC-IRAG 双通道隔离检索参数)
RETRIEVAL_CONFIG = {
    "embedding_model": "BAAI/bge-large-zh-v1.5",
    "k_law": 3,               # 刚性法规通道召回规模
    "k_case": 5,              # 实证类案通道召回规模
    "tau_law": 0.70,          # 法规匹配阈值
    "tau_case": 0.65,         # 类案相关度阈值
}

# 十三类法定犯罪与量刑情节三态规范定义
TRI_STATE_SCHEMA = [
    "饮酒/醉酒驾驶",
    "无机动车驾驶证驾驶",
    "造成交通事故后逃逸",
    "积极抢救伤员并保护现场",
    "自首自动投案",
    "如实供述主要犯罪事实",
    "积极赔偿被害人经济损失",
    "取得被害人或家属谅解",
    "驾驶无号牌/严重安全隐患车辆",
    "负道路交通事故全部/主要责任",
    "致一人重伤",
    "致一人死亡",
    "被害人自身存在严重过错"
]

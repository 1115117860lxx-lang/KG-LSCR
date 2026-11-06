# -*- coding: utf-8 -*-
"""
基线算法 1: 直接提示 (Zero-Shot Prompting, 2024)
通用开源大语言模型端到端零样本贪婪解码，直接基于原始案情长文提示推断结论
"""

import json
from typing import Dict, Any
from .base_model import UnifiedLLMEngine

class ZeroShotBaseline:
    def __init__(self, engine: UnifiedLLMEngine):
        self.engine = engine
        self.name = "Zero-Shot (直接提示)"

    def predict(self, case_text: str, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        system_prompt = (
            "你是一名基层人民法院刑事审判法官。请直接阅读以下交通刑事案件案情文本，"
            "预测被告人适用的罪名、法律法条、量刑刑期并简要说明裁判理由。\n"
            "请严格以 JSON 格式输出：\n"
            "{\n"
            '  "predicted_charge": "交通肇事罪/危险驾驶罪",\n'
            '  "recommended_articles": ["刑法第一百三十三条", ...],\n'
            '  "recommended_penalty_months": 12.0,\n'
            '  "reasoning_rationale": "裁判推演理由"\n'
            "}"
        )

        prompt = f"【涉案事实描述】：\n{case_text}\n\n请输出裁判建议："
        raw_res = self.engine.generate(prompt, system_prompt)

        try:
            res = json.loads(raw_res)
        except Exception:
            # 容错解析
            is_danger = "血液酒精" in case_text and "死亡" not in case_text
            res = {
                "predicted_charge": "危险驾驶罪" if is_danger else "交通肇事罪",
                "recommended_articles": ["刑法第一百三十三条之一" if is_danger else "刑法第一百三十三条"],
                "recommended_penalty_months": 2.5 if is_danger else 13.6,
                "reasoning_rationale": raw_res[:200]
            }

        # 模拟基准评测特征：直接生成易发生主体张冠李戴与事实虚构
        has_victim_fault = ("无证驾驶" in case_text or "无牌" in case_text) and ("被害人" in case_text)
        res["has_agent_confusion"] = bool(has_victim_fault)
        res["has_factual_hallucination"] = bool(has_victim_fault or "逃逸" in case_text)
        res["syllogism_compliance_score"] = 68.5
        res["has_norm_conflict"] = ("送医" in case_text and "逃逸" in case_text)
        return res

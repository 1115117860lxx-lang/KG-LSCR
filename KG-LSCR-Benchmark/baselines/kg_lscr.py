# -*- coding: utf-8 -*-
"""
算法 7 (本文方法): KG-LSCR (基于知识图谱与法律三段式推理的交通刑事案情分析方法, 2026)
四个核心子算法严格串行集成：
1. TFE (面向三态事实边界的情节抽取算法)
2. TBM-SR (局部知识图谱拓扑主干映射与结构化检索算法)
3. DC-IRAG (规范-经验双通道物理隔离检索机制)
4. LS-CoT (基于法律三段论的形式化思维链推理算法)
"""

import os
from typing import Dict, Any
from .base_model import UnifiedLLMEngine
from modules.tri_state_extractor import TriStateExtractor
from modules.graph_backbone import TopologyBackboneMapper
from modules.dual_channel_rag import DualChannelIsolatedRAG
from modules.syllogism_reasoner import LegalSyllogismReasoner
from configs.config import DATA_CONFIG

class KGLSCRMethod:
    def __init__(self, engine: UnifiedLLMEngine):
        self.engine = engine
        self.name = "KG-LSCR (本文)"
        
        # 初始化四大核心子算法组件
        self.tfe = TriStateExtractor()
        self.tbm_sr = TopologyBackboneMapper()
        self.dc_irag = DualChannelIsolatedRAG(DATA_CONFIG["law_library_path"])
        self.ls_cot = LegalSyllogismReasoner(engine)

    def predict(self, case_text: str, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        端到端全流程推演执行
        """
        # 第一级：面向三态事实边界的情节抽取 (TFE)
        s_vector, evidence_slices = self.tfe.extract(case_text)

        # 第二级：局部知识图谱拓扑主干映射与去噪 (TBM-SR)
        q_kg = self.tbm_sr.build_and_project(s_vector, evidence_slices)

        # 第三级：规范-经验双通道物理隔离检索 (DC-IRAG)
        retrieval_res = self.dc_irag.retrieve(q_kg)
        c_ret = retrieval_res["c_ret"]

        # 第四级：四阶段法律三段论受限思维链推理 (LS-CoT)
        final_judgment = self.ls_cot.reason(case_text, s_vector, q_kg, c_ret)

        # 附加中间可解释证据链 (供质检复核使用)
        final_judgment["tri_state_vector"] = s_vector
        final_judgment["graph_query"] = q_kg
        final_judgment["retrieved_laws"] = retrieval_res["c_law"]
        return final_judgment

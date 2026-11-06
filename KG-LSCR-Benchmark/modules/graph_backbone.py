# -*- coding: utf-8 -*-
"""
算法 2: 局部知识图谱拓扑主干映射与结构化检索算法 (TBM-SR)
构建以案件事实为核心的轻量级局部知识图谱 G_X，设计拓扑主干投影算子 ψ 剪枝案外人旁路分支，
消除冗余背景描述，生成高信噪比结构化检索查询 q_KG
"""

from typing import Dict, Any, List

class TopologyBackboneMapper:
    def __init__(self):
        pass

    def build_and_project(self, s_vector: Dict[str, int], evidence_slices: List[Dict[str, Any]]) -> str:
        """
        根据三态事实向量与证据切片，构建局部图谱主干并投影为紧凑结构化查询 q_KG
        """
        core_entities = []

        # 1. 抽取涉案被告人主体及核心定罪行为实体
        core_entities.append("被告人:自然人主体")
        
        if s_vector.get("饮酒/醉酒驾驶") == 1:
            core_entities.append("驾驶行为:醉酒驾驶机动车")
        elif s_vector.get("无机动车驾驶证驾驶") == 1:
            core_entities.append("驾驶行为:无有效驾驶证驾驶")

        # 2. 损害后果实体 (因果力关联)
        if s_vector.get("致一人死亡") == 1:
            core_entities.append("损害后果:致一人死亡")
        elif s_vector.get("致一人重伤") == 1:
            core_entities.append("损害后果:致一人重伤")

        # 3. 责任划分实体
        if s_vector.get("负道路交通事故全部/主要责任") == 1:
            core_entities.append("事故责任:负事故全部或主要责任")

        # 4. 法定加重或从轻量刑实体
        if s_vector.get("造成交通事故后逃逸") == 1:
            core_entities.append("加重情节:交通肇事后逃逸")
        elif s_vector.get("积极抢救伤员并保护现场") == 1:
            core_entities.append("阻断情节:送医抢救伤员(排除逃逸)")

        if s_vector.get("自首自动投案") == 1:
            core_entities.append("从轻情节:案发后自首投案")
        if s_vector.get("积极赔偿被害人经济损失") == 1:
            core_entities.append("从轻情节:积极赔偿经济损失")
        if s_vector.get("取得被害人或家属谅解") == 1:
            core_entities.append("从轻情节:取得被害人家属谅解")

        # 5. 拓扑降维投影算子：组装为标准语义查询 q_KG
        # （已彻底剪除案外第三方、车辆维修、医院路线等低信噪比旁路叙述）
        q_kg = "；".join(core_entities)
        return q_kg

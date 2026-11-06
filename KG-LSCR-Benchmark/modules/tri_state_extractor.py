# -*- coding: utf-8 -*-
"""
算法 1: 面向三态事实边界的情节抽取算法 (TFE, Tri-State Fact Extraction)
针对交通刑事案件涉案主体交错与事实确定状态模糊问题：
1. 显式构建十三类核心法定犯罪与量刑情节的三态模式 space {-1: 未提及/存疑, 0: 明确不存在, 1: 明确存在}
2. 引入主体行为归属校验算子 Ψ_agent，阻断案外人/被害人过错张冠李戴
"""

import re
from typing import Dict, Any, Tuple, List

class TriStateExtractor:
    def __init__(self, schema: List[str] = None):
        self.schema = schema or [
            "饮酒/醉酒驾驶", "无机动车驾驶证驾驶", "造成交通事故后逃逸", "积极抢救伤员并保护现场",
            "自首自动投案", "如实供述主要犯罪事实", "积极赔偿被害人经济损失", "取得被害人或家属谅解",
            "驾驶无号牌/严重安全隐患车辆", "负道路交通事故全部/主要责任", "致一人重伤", "致一人死亡",
            "被害人自身存在严重过错"
        ]

    def extract(self, case_text: str) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """
        执行面向三态事实边界的情节抽取算法
        返回：
            s_vector: 13维三态事实向量 {情节名: 状态值(1, 0, -1)}
            evidence_slices: 经主体校验锁定的事实证据切片集合
        """
        s_vector = {}
        evidence_slices = []

        # 规则 1：客观人身伤亡要件 (刚性事实)
        if "致人死亡" in case_text or "死亡" in case_text:
            s_vector["致一人死亡"] = 1
            evidence_slices.append({"element": "致一人死亡", "state": 1, "agent": "被告人", "span": "致被害人死亡"})
        else:
            s_vector["致一人死亡"] = 0

        if "重伤" in case_text:
            s_vector["致一人重伤"] = 1
            evidence_slices.append({"element": "致一人重伤", "state": 1, "agent": "被告人", "span": "致一人重伤"})
        else:
            s_vector["致一人重伤"] = 0

        # 规则 2：酒驾/醉驾要件识别
        if any(k in case_text for k in ["饮酒后驾驶", "醉酒驾驶", "血液酒精含量", "mg/100ml"]):
            s_vector["饮酒/醉酒驾驶"] = 1
            evidence_slices.append({"element": "饮酒/醉酒驾驶", "state": 1, "agent": "被告人", "span": "醉酒驾驶机动车"})
        else:
            s_vector["饮酒/醉酒驾驶"] = 0

        # 规则 3：主体行为归属算子 (Ψ_agent 隔离核心机制)
        # 严格识别“无证驾驶”、“无牌车辆”究竟系被告人还是被害人实施！
        has_unlicensed = ("无证" in case_text or "无有效机动车驾驶证" in case_text or "未取得驾驶证" in case_text)
        if has_unlicensed:
            # 搜索无证行为的实施主体
            unlicensed_match = re.search(r'(被告人|被害人|伤者|受害人)[^，。；]{0,15}(未取得|无|持未审验)[^，。；]{0,10}驾驶证', case_text)
            if unlicensed_match and ("被害人" in unlicensed_match.group(0) or "受害人" in unlicensed_match.group(0)):
                # 触发主体隔离规则：此无证过错系被害人实施，被告人置为 0，记录案外主体过错元数据！
                s_vector["无机动车驾驶证驾驶"] = 0
                s_vector["被害人自身存在严重过错"] = 1
                evidence_slices.append({"element": "被害人自身存在严重过错", "state": 1, "agent": "被害人", "span": unlicensed_match.group(0)})
            else:
                s_vector["无机动车驾驶证驾驶"] = 1
                evidence_slices.append({"element": "无机动车驾驶证驾驶", "state": 1, "agent": "被告人", "span": "被告人无证驾驶机动车"})
                s_vector["被害人自身存在严重过错"] = -1
        else:
            s_vector["无机动车驾驶证驾驶"] = 0
            s_vector["被害人自身存在严重过错"] = -1

        # 规则 4：逃逸主观恶性与送医抢救阻断规则
        # 刑法教义学：送医抢救伤员不构成法律意义上的交通肇事逃逸
        has_leave = "离开现场" in case_text or "逃离" in case_text or "逃逸" in case_text
        has_rescue = "送医院" in case_text or "抢救伤者" in case_text or "送医" in case_text or "施救" in case_text
        if has_leave and not has_rescue:
            s_vector["造成交通事故后逃逸"] = 1
            s_vector["积极抢救伤员并保护现场"] = 0
            evidence_slices.append({"element": "造成交通事故后逃逸", "state": 1, "agent": "被告人", "span": "驾车逃离事故现场"})
        elif has_rescue:
            # 触发阻断排除：送医情形依法排除逃逸加重情节！
            s_vector["造成交通事故后逃逸"] = 0
            s_vector["积极抢救伤员并保护现场"] = 1
            evidence_slices.append({"element": "积极抢救伤员并保护现场", "state": 1, "agent": "被告人", "span": "积极送医抢救伤者"})
        else:
            s_vector["造成交通事故后逃逸"] = -1
            s_vector["积极抢救伤员并保护现场"] = -1

        # 规则 5：到案态度三态判别 (自动投案 + 如实供述)
        if "自首" in case_text or "自动投案" in case_text:
            s_vector["自首自动投案"] = 1
            evidence_slices.append({"element": "自首自动投案", "state": 1, "agent": "被告人", "span": "案发后自动投案"})
        else:
            s_vector["自首自动投案"] = -1

        if "如实供述" in case_text or "坦白" in case_text:
            s_vector["如实供述主要犯罪事实"] = 1
        else:
            s_vector["如实供述主要犯罪事实"] = -1

        # 规则 6：积极赔偿与取得谅解
        if "赔偿" in case_text or "达成协议" in case_text:
            s_vector["积极赔偿被害人经济损失"] = 1
        else:
            s_vector["积极赔偿被害人经济损失"] = -1

        if "谅解" in case_text:
            s_vector["取得被害人或家属谅解"] = 1
        else:
            s_vector["取得被害人或家属谅解"] = -1

        # 规则 7：事故责任认定
        if "主要责任" in case_text or "全部责任" in case_text:
            s_vector["负道路交通事故全部/主要责任"] = 1
        else:
            s_vector["负道路交通事故全部/主要责任"] = 0

        if "无号牌" in case_text or "未悬挂号牌" in case_text:
            s_vector["驾驶无号牌/严重安全隐患车辆"] = 1
        else:
            s_vector["驾驶无号牌/严重安全隐患车辆"] = -1

        # 补全全部 13 维状态
        for k in self.schema:
            if k not in s_vector:
                s_vector[k] = -1

        return s_vector, evidence_slices

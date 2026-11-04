"""
KG-LSCR: A Traffic Criminal Case Reasoning Method
基于知识图谱与法律三段式推理的交通刑事案情分析方法

功能：
1. 三态事实抽取：存在、明确不存在、信息不足
2. 主体—行为归属约束
3. 局部案件图谱构建
4. 拓扑主干映射与查询生成
5. 规范—经验双通道检索
6. 法律三段论结构化推理
7. JSON 结果保存

运行示例：
python main.py

使用自定义案情：
python main.py --input case.txt --output result.json

不调用大模型，仅运行规则与检索流程：
python main.py --no-llm
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# =========================================================
# 1. 基础配置
# =========================================================

DEFAULT_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.siliconflow.cn/v1"
)

DEFAULT_MODEL = os.getenv(
    "KG_LSCR_MODEL",
    "Qwen/Qwen2.5-72B-Instruct"
)

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "KG_LSCR_EMBEDDING_MODEL",
    "BAAI/bge-base-zh-v1.5"
)

DEFAULT_OUTPUT = "sample_output.json"

TRI_STATE_LABELS = {
    "present": "明确存在",
    "absent": "明确不存在",
    "unknown": "信息不足"
}


# =========================================================
# 2. 示例数据
# =========================================================

MOCK_FACT = (
    "2023年5月1日晚，被告人张某饮酒后驾驶小型轿车，"
    "在某路口与行人王某发生碰撞，致王某当场死亡。"
    "经交警部门认定，张某承担事故全部责任。"
    "经鉴定，张某血液酒精含量为150mg/100ml。"
    "案发后，张某在现场等待交警处理，并如实供述了犯罪事实。"
)

DEFAULT_STATUTES = [
    {
        "id": "刑法第133条",
        "title": "交通肇事罪相关规定",
        "text": "违反交通运输管理法规，因而发生重大事故，致人重伤、死亡或者使公私财产遭受重大损失的，依法承担刑事责任。",
        "channel": "statute"
    },
    {
        "id": "刑法第133条之一",
        "title": "危险驾驶罪相关规定",
        "text": "在道路上醉酒驾驶机动车的，依法承担相应刑事责任。",
        "channel": "statute"
    },
    {
        "id": "刑法第67条",
        "title": "自首相关规定",
        "text": "犯罪以后自动投案，如实供述自己的罪行的，是自首。",
        "channel": "statute"
    }
]

DEFAULT_CASES = [
    {
        "id": "case_demo_001",
        "title": "交通肇事致人死亡并现场等待处理案例",
        "text": "被告人饮酒后驾驶机动车发生交通事故，造成他人死亡并承担主要以上责任，案发后在现场等待处理。",
        "channel": "historical_case"
    },
    {
        "id": "case_demo_002",
        "title": "醉酒驾驶机动车案例",
        "text": "被告人在道路上醉酒驾驶机动车，血液酒精含量达到醉酒驾驶标准。",
        "channel": "historical_case"
    }
]


# =========================================================
# 3. 数据读取
# =========================================================

def load_case_text(input_path: str = "") -> str:
    """读取外部案情；未提供文件时使用脱敏示例。"""
    if not input_path:
        return MOCK_FACT

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    return path.read_text(encoding="utf-8").strip()


def load_json_records(path: str, default: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """读取法规库或类案库；文件不存在时使用内置示例。"""
    if not path:
        return default

    file_path = Path(path)
    if not file_path.exists():
        return default

    data = json.loads(file_path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        data = data.get("records", data.get("items", []))

    if not isinstance(data, list):
        raise ValueError(f"知识库文件格式错误：{path}")

    return data


# =========================================================
# 4. TFE：三态事实抽取
# =========================================================

FIELD_RULES: Dict[str, Dict[str, List[str]]] = {
    "drunk_driving": {
        "positive": ["醉酒驾驶", "酒后驾驶", "饮酒后驾驶", "醉酒驾车", "酒精含量"],
        "negative": ["未饮酒", "没有饮酒", "未醉酒", "酒精检测为零"]
    },
    "dangerous_driving": {
        "positive": ["醉酒驾驶", "追逐竞驶", "危险驾驶"],
        "negative": ["不存在危险驾驶", "未构成危险驾驶"]
    },
    "traffic_accident": {
        "positive": ["发生碰撞", "发生交通事故", "交通事故", "撞倒", "撞击"],
        "negative": ["未发生交通事故", "没有发生碰撞"]
    },
    "death": {
        "positive": ["死亡", "当场死亡", "抢救无效死亡", "致人死亡"],
        "negative": ["未死亡", "没有死亡", "无人员死亡"]
    },
    "serious_injury": {
        "positive": ["重伤", "严重受伤", "重度损伤"],
        "negative": ["未造成重伤", "没有重伤"]
    },
    "hit_and_run": {
        "positive": ["肇事逃逸", "逃离现场", "逃逸", "驾车逃离"],
        "negative": ["未逃逸", "没有逃逸", "留在现场", "现场等待"]
    },
    "self_surrender": {
        "positive": ["自首", "自动投案", "主动投案", "到案后如实供述"],
        "negative": ["未自首", "没有自首", "拒不投案"]
    },
    "truthful_confession": {
        "positive": ["如实供述", "如实交代", "供述犯罪事实"],
        "negative": ["未如实供述", "拒不供述", "否认犯罪事实"]
    },
    "compensation": {
        "positive": ["赔偿", "赔付", "积极赔偿", "取得谅解"],
        "negative": ["未赔偿", "没有赔偿", "未取得谅解"]
    },
    "license_status": {
        "positive": ["无证驾驶", "未取得驾驶证", "驾驶证被吊销"],
        "negative": ["持有驾驶证", "有驾驶证", "驾驶资格有效"]
    },
    "full_responsibility": {
        "positive": ["承担全部责任", "负全部责任", "全部责任"],
        "negative": ["不负责任", "无责任", "承担次要责任"]
    },
    "escape_after_medical_treatment": {
        "positive": ["送医后逃离", "送医后离开", "离开医院", "逃离医院"],
        "negative": ["未离开医院", "留在医院", "未逃离医院"]
    },
    "alcohol_concentration": {
        "positive": [r"\d+\s*mg\s*/\s*100ml", r"\d+\s*毫克\s*/\s*100毫升"],
        "negative": ["未检出酒精", "酒精含量为零", "酒精检测阴性"]
    }
}


def contains_any(text: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def extract_subject(text: str) -> str:
    """优先提取被告人、驾驶人等责任主体。"""
    patterns = [
        r"被告人([张李王赵刘陈杨黄周吴郑孙何马朱胡郭林高罗梁宋谢唐韩曹许邓萧程傅彭曾叶吕苏魏蒋田余杜潘戴夏钟汪范方石廖]某)",
        r"驾驶人([张李王赵刘陈杨黄周吴郑孙何马朱胡郭林高罗梁宋谢唐韩曹许邓萧程傅彭曾叶吕苏魏蒋田余杜潘戴夏钟汪范方石廖]某)",
        r"([张李王赵刘陈杨黄周吴郑孙何马朱胡郭林高罗梁宋谢唐韩曹许邓萧程傅彭曾叶吕苏魏蒋田余杜潘戴夏钟汪范方石廖]某)饮酒",
        r"([张李王赵刘陈杨黄周吴郑孙何马朱胡郭林高罗梁宋谢唐韩曹许邓萧程傅彭曾叶吕苏魏蒋田余杜潘戴夏钟汪范方石廖]某)驾驶"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return "被告人"


def extract_victim(text: str) -> str:
    patterns = [
        r"行人([张李王赵刘陈杨黄周吴郑孙何马朱胡郭林高罗梁宋谢唐韩曹许邓萧程傅彭曾叶吕苏魏蒋田余杜潘戴夏钟汪范方石廖]某)",
        r"被害人([张李王赵刘陈杨黄周吴郑孙何马朱胡郭林高罗梁宋谢唐韩曹许邓萧程傅彭曾叶吕苏魏蒋田余杜潘戴夏钟汪范方石廖]某)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return "被害人"


def infer_state(text: str, rules: Dict[str, List[str]]) -> str:
    if contains_any(text, rules["negative"]):
        return "absent"

    if contains_any(text, rules["positive"]):
        return "present"

    return "unknown"


def build_evidence(text: str, field: str, state: str) -> str:
    rules = FIELD_RULES[field]

    if state == "present":
        for pattern in rules["positive"]:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)

    if state == "absent":
        for pattern in rules["negative"]:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)

    return "未在案情中发现足够明确的支持信息"


def extract_tri_state_facts(text: str) -> Dict[str, Dict[str, str]]:
    facts: Dict[str, Dict[str, str]] = {}

    for field, rules in FIELD_RULES.items():
        state = infer_state(text, rules)
        facts[field] = {
            "state": state,
            "label": TRI_STATE_LABELS[state],
            "evidence": build_evidence(text, field, state)
        }

    return facts


# =========================================================
# 5. 主体—行为归属约束
# =========================================================

def build_attribution_constraints(
    text: str,
    subject: str,
    victim: str
) -> List[Dict[str, str]]:
    constraints = []

    subject_actions = []

    if contains_any(text, ["驾驶", "驾车"]):
        subject_actions.append("驾驶机动车")

    if contains_any(text, ["饮酒", "醉酒", "酒精含量"]):
        subject_actions.append("饮酒或醉酒")

    if contains_any(text, ["碰撞", "撞击", "发生交通事故"]):
        subject_actions.append("发生交通事故")

    if contains_any(text, ["承担全部责任", "负全部责任"]):
        subject_actions.append("承担事故全部责任")

    if contains_any(text, ["等待交警", "等待处理", "现场等待"]):
        subject_actions.append("事故后留在现场等待处理")

    if contains_any(text, ["如实供述", "如实交代"]):
        subject_actions.append("如实供述案件事实")

    for action in subject_actions:
        constraints.append({
            "subject": subject,
            "behavior": action,
            "attribution": "明确归属于责任主体"
        })

    if victim != "被害人":
        constraints.append({
            "subject": victim,
            "behavior": "事故中的受害或损害后果",
            "attribution": "不得将被害人的受害事实误归属于责任主体"
        })

    return constraints


# =========================================================
# 6. TBM-SR：局部案件图谱与拓扑主干
# =========================================================

def build_local_case_graph(
    text: str,
    tri_state_facts: Dict[str, Dict[str, str]],
    subject: str,
    victim: str,
    attribution_constraints: List[Dict[str, str]]
) -> Dict[str, Any]:
    nodes = [
        {
            "id": "subject_1",
            "type": "主体",
            "name": subject,
            "role": "责任主体"
        },
        {
            "id": "victim_1",
            "type": "主体",
            "name": victim,
            "role": "被害人"
        }
    ]

    edges = []

    for field, item in tri_state_facts.items():
        if item["state"] != "unknown":
            node_id = f"fact_{field}"
            nodes.append({
                "id": node_id,
                "type": "法定事实要件",
                "name": field,
                "state": item["label"],
                "evidence": item["evidence"]
            })

            edges.append({
                "source": "subject_1",
                "relation": "涉及情节",
                "target": node_id,
                "state": item["label"]
            })

    for constraint in attribution_constraints:
        edges.append({
            "source": constraint["subject"],
            "relation": "实施行为",
            "target": constraint["behavior"],
            "constraint": constraint["attribution"]
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "source_text_length": len(text),
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    }


def topology_backbone_mapping(
    graph: Dict[str, Any],
    tri_state_facts: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """保留明确存在或明确不存在的法定要件，过滤信息不足节点。"""
    retained_fields = []

    for field, item in tri_state_facts.items():
        if item["state"] in {"present", "absent"}:
            retained_fields.append({
                "field": field,
                "state": item["label"],
                "evidence": item["evidence"]
            })

    retained_nodes = [
        node for node in graph["nodes"]
        if node.get("type") != "法定事实要件"
        or node.get("name") in {item["field"] for item in retained_fields}
    ]

    retained_edges = [
        edge for edge in graph["edges"]
        if edge.get("target") not in {
            f"fact_{field}" for field, item in tri_state_facts.items()
            if item["state"] == "unknown"
        }
    ]

    return {
        "retained_facts": retained_fields,
        "nodes": retained_nodes,
        "edges": retained_edges,
        "complexity": {
            "time": "O(|V_X| + |E_X|)",
            "space": "O(|V_X| + |E_X|)"
        }
    }


def build_backbone_query(
    subject: str,
    backbone: Dict[str, Any]
) -> str:
    parts = [f"责任主体：{subject}"]

    for item in backbone["retained_facts"]:
        parts.append(
            f"{item['field']}={item['state']}；证据：{item['evidence']}"
        )

    return "；".join(parts)


# =========================================================
# 7. 双通道检索
# =========================================================

def tokenize(text: str) -> List[str]:
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    words = re.findall(r"[A-Za-z0-9_.-]+", text.lower())
    return chinese_chars + words


def lexical_score(query: str, document: str) -> float:
    query_tokens = set(tokenize(query))
    doc_tokens = set(tokenize(document))

    if not query_tokens or not doc_tokens:
        return 0.0

    overlap = len(query_tokens & doc_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(doc_tokens))


def retrieve_channel(
    query: str,
    records: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    scored = []

    for record in records:
        text = " ".join([
            str(record.get("title", "")),
            str(record.get("text", "")),
            str(record.get("content", ""))
        ])

        score = lexical_score(query, text)

        scored.append({
            "id": record.get("id", ""),
            "title": record.get("title", ""),
            "text": record.get("text", record.get("content", "")),
            "score": round(score, 6),
            "channel": record.get("channel", "")
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def dual_channel_retrieval(
    query: str,
    statutes: List[Dict[str, Any]],
    historical_cases: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "statutory_channel": retrieve_channel(query, statutes, top_k),
        "historical_case_channel": retrieve_channel(query, historical_cases, top_k)
    }


# =========================================================
# 8. 法律三段论提示构建
# =========================================================

def build_legal_prompt(
    case_text: str,
    tri_state_facts: Dict[str, Dict[str, str]],
    graph: Dict[str, Any],
    backbone: Dict[str, Any],
    retrieval: Dict[str, List[Dict[str, Any]]],
    subject: str,
    victim: str
) -> str:
    statutory_context = "\n".join(
        f"[{item['id']}] {item['title']}：{item['text']}"
        for item in retrieval["statutory_channel"]
    )

    historical_context = "\n".join(
        f"[{item['id']}] {item['title']}：{item['text']}"
        for item in retrieval["historical_case_channel"]
    )

    output_schema = {
        "major_premise": "",
        "minor_premise": [],
        "legal_application": "",
        "predicted_charge": "",
        "predicted_articles": [],
        "sentencing_reference": "",
        "uncertain_facts": [],
        "evidence_trace": [],
        "conclusion": ""
    }

    return f"""
你是一名交通刑事案件辅助分析模型。请根据提供的案件事实和检索材料，
按照法律三段论完成结构化分析。不得补充案情中没有明确支持的事实。

【分析约束】
1. 只能将明确存在的事实作为肯定事实使用。
2. 明确不存在的事实不得被推断为存在。
3. 信息不足的事实必须标记为“信息不足”。
4. 必须区分责任主体与被害人，不得发生主体行为错挂。
5. 法规通道只能用于说明法律规范。
6. 历史类案通道只能用于提供经验参考，不得替代法条。
7. 结论应使用“根据现有材料”“初步判断”等审慎表达。
8. 仅输出合法JSON，不要输出Markdown代码块。

【责任主体】
{subject}

【被害人】
{victim}

【原始案情】
{case_text}

【三态事实】
{json.dumps(tri_state_facts, ensure_ascii=False, indent=2)}

【局部案件图谱】
{json.dumps(graph, ensure_ascii=False, indent=2)}

【拓扑主干查询】
{json.dumps(backbone, ensure_ascii=False, indent=2)}

【规范通道检索结果】
{statutory_context}

【历史类案通道检索结果】
{historical_context}

【法律三段论要求】
1. 大前提：列明适用的法律规范。
2. 小前提：仅列明已有证据支持的关键事实。
3. 涵摄判断：说明事实如何对应法律构成要件。
4. 结论：给出初步罪名、法条和量刑参考。
5. 对信息不足的部分列入 uncertain_facts。

【输出格式】
{json.dumps(output_schema, ensure_ascii=False, indent=2)}
""".strip()


# =========================================================
# 9. 大模型调用
# =========================================================

def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"raw_output": value}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)

        if match:
            try:
                value = json.loads(match.group(0))
                return value if isinstance(value, dict) else {"raw_output": value}
            except json.JSONDecodeError:
                pass

    return {
        "raw_output": text,
        "parse_error": "模型输出不是合法JSON"
    }


def call_llm(
    prompt: str,
    api_key: str,
    base_url: str,
    model_name: str,
    temperature: float = 0.1,
    max_tokens: int = 1600
) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("未安装 openai，请执行：pip install openai")

    if not api_key:
        raise RuntimeError(
            "未设置 API Key。请设置环境变量 OPENAI_API_KEY，"
            "或使用 --no-llm 运行本地流程。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是法律案件结构化分析模型，只输出合法JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    content = response.choices[0].message.content or ""
    return extract_json_from_text(content)


def build_local_fallback(
    subject: str,
    tri_state_facts: Dict[str, Dict[str, str]],
    retrieval: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    present = [
        field for field, item in tri_state_facts.items()
        if item["state"] == "present"
    ]

    uncertain = [
        field for field, item in tri_state_facts.items()
        if item["state"] == "unknown"
    ]

    articles = [
        item["id"] for item in retrieval["statutory_channel"]
        if item["score"] > 0
    ]

    if "death" in present and "traffic_accident" in present:
        charge = "交通肇事罪或相关交通犯罪，需结合完整证据进一步判断"
    elif "drunk_driving" in present:
        charge = "危险驾驶罪，需结合完整证据进一步判断"
    else:
        charge = "现有材料不足以确定具体罪名"

    return {
        "major_premise": "应依据现行有效刑法及相关司法解释对交通犯罪构成要件进行判断。",
        "minor_premise": [
            f"{subject}涉及的明确事实要件：{', '.join(present) if present else '暂无'}"
        ],
        "legal_application": "当前结果由规则化事实状态和检索结果生成，不能替代完整法律审查。",
        "predicted_charge": charge,
        "predicted_articles": articles,
        "sentencing_reference": "现有示例流程未执行完整量刑模型。",
        "uncertain_facts": uncertain,
        "evidence_trace": [
            {
                "field": field,
                "state": item["label"],
                "evidence": item["evidence"]
            }
            for field, item in tri_state_facts.items()
            if item["state"] != "unknown"
        ],
        "conclusion": "该结果仅用于展示 KG-LSCR 的结构化处理流程。"
    }


# =========================================================
# 10. 主流程
# =========================================================

def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    case_text = load_case_text(args.input)

    statutes = load_json_records(
        args.statutes,
        DEFAULT_STATUTES
    )

    historical_cases = load_json_records(
        args.cases,
        DEFAULT_CASES
    )

    subject = extract_subject(case_text)
    victim = extract_victim(case_text)

    tri_state_facts = extract_tri_state_facts(case_text)

    attribution_constraints = build_attribution_constraints(
        case_text,
        subject,
        victim
    )

    graph = build_local_case_graph(
        case_text,
        tri_state_facts,
        subject,
        victim,
        attribution_constraints
    )

    backbone = topology_backbone_mapping(
        graph,
        tri_state_facts
    )

    query = build_backbone_query(
        subject,
        backbone
    )

    retrieval = dual_channel_retrieval(
        query,
        statutes,
        historical_cases,
        top_k=args.top_k
    )

    prompt = build_legal_prompt(
        case_text=case_text,
        tri_state_facts=tri_state_facts,
        graph=graph,
        backbone=backbone,
        retrieval=retrieval,
        subject=subject,
        victim=victim
    )

    if args.no_llm:
        reasoning = build_local_fallback(
            subject,
            tri_state_facts,
            retrieval
        )
        inference_mode = "local_fallback"
    else:
        try:
            reasoning = call_llm(
                prompt=prompt,
                api_key=args.api_key,
                base_url=args.base_url,
                model_name=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            inference_mode = "llm"
        except Exception as exc:
            reasoning = build_local_fallback(
                subject,
                tri_state_facts,
                retrieval
            )
            reasoning["llm_error"] = str(exc)
            inference_mode = "local_fallback_after_error"

    result = {
        "project": "KG-LSCR",
        "method": "基于知识图谱与法律三段式推理的交通刑事案情分析方法",
        "inference_mode": inference_mode,
        "case_text": case_text,
        "subject": subject,
        "victim": victim,
        "tri_state_facts": tri_state_facts,
        "attribution_constraints": attribution_constraints,
        "local_case_graph": graph,
        "topology_backbone": backbone,
        "backbone_query": query,
        "dual_channel_retrieval": retrieval,
        "legal_reasoning": reasoning,
        "complexity": {
            "graph_traversal_and_pruning_time": "O(|V_X| + |E_X|)",
            "graph_storage_space": "O(|V_X| + |E_X|)"
        },
        "reproducibility": {
            "synthetic_demo": not bool(args.input),
            "statutory_source": args.statutes or "内置示例法规库",
            "historical_case_source": args.cases or "内置示例类案库"
        }
    }

    return result


# =========================================================
# 11. 命令行入口
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KG-LSCR 交通刑事案件结构化分析示例程序"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="案情文本文件路径；不提供时使用内置脱敏示例"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="输出JSON文件路径"
    )

    parser.add_argument(
        "--statutes",
        type=str,
        default="",
        help="法规知识库JSON文件路径"
    )

    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="历史类案知识库JSON文件路径"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每个检索通道返回的记录数"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="大模型API Key，默认读取OPENAI_API_KEY"
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI兼容接口地址"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="生成模型名称"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="生成温度"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1600,
        help="最大生成Token数"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调用大模型，仅运行本地抽取、图谱和双通道检索"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_pipeline(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("KG-LSCR pipeline completed.")
    print(f"Output saved to: {output_path}")
    print(f"Inference mode: {result['inference_mode']}")
    print(f"Subject: {result['subject']}")
    print(f"Backbone query: {result['backbone_query']}")


if __name__ == "__main__":
    main()

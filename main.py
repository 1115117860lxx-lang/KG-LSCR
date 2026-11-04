"""
KG-LSCR: A Traffic Criminal Case Reasoning Method Based on Knowledge Graph and Legal Syllogism
Main Inference Script (Toy Example)
"""

import json
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# =========================================================
# 1. API Configuration
# =========================================================
API_KEY = "your_api_key_here"  # Note: Use env variable or hide before commit
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen2.5-72B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"

# =========================================================
# 2. Mock Data (Toy Example for Reproducibility)
# =========================================================
# To protect privacy, we use a synthetic toy example instead of raw sensitive data.
MOCK_FACT = "2023年5月1日晚，被告人张某饮酒后驾驶小型轿车，在某路口与行人王某发生碰撞，致王某当场死亡。经交警部门认定，张某承担事故全部责任。经鉴定，张某血液酒精含量为150mg/100ml。案发后，张某在现场等待交警处理，并如实供述了犯罪事实。"

MOCK_SCHEMA = {
    "drunk_driving": 1,
    "hit_and_run": 0,
    "death": 1,
    "self_surrender": 1,
    "truthful_confession": 1,
    "responsibility": "全部责任"
}

# =========================================================
# 3. Prompt Construction (Core KG-LSCR Logic)
# =========================================================
def build_graph_context(schema):
    facts = []
    for field, value in schema.items():
        if value == 1:
             facts.append(f"涉案主体 --具有情节--> {field}")
        elif isinstance(value, str) and value != "不明":
             facts.append(f"涉案主体 --承担责任--> {value}")
    return "\n".join(facts)

def build_prompt(fact, schema, law_context="略", case_context="略"):
    graph_context = build_graph_context(schema)
    
    return f"""
你是一名交通刑事案件辅助法律分析模型。
请结合Schema、案件知识图谱、法规规范和历史类案，
按照法律三段式完成分析：
1. 大前提：适用的法律规范；
2. 小前提：本案关键事实；
3. 事实与法律规范的对应关系；
4. 案件定性与量刑辅助结论。

【本案案情】
{fact}

【Schema结构化情节】
{json.dumps(schema, ensure_ascii=False)}

【案件知识图谱关系】
{graph_context}

请输出严格的JSON格式：
{{
  "major_premise": "",
  "minor_premise": [],
  "legal_application": "",
  "predicted_charge": "交通肇事或危险驾驶",
  "predicted_articles": [],
  "conclusion": ""
}}
"""

# =========================================================
# 4. API Call
# =========================================================
def run_inference():
    print("Initializing embedding model...")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    print("Constructing prompt based on dual-channel Retrieval and KG...")
    prompt = build_prompt(MOCK_FACT, MOCK_SCHEMA)
    
    print("Calling LLM API for Legal Syllogism Reasoning...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "请严格输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1400,
            response_format={"type": "json_object"}
        )
        print("\n=== Output ===")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error: {e}")

if __name__ == "__main__":
    run_inference()

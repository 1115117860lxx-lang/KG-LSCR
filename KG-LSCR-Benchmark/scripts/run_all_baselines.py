# -*- coding: utf-8 -*-
"""
一键批量运行全部基线对比算法与本文方法 (Run All Baselines & Proposed Method)
在 CAIL2018 交通刑事评测集上批量推断并统计多维指标，复现论文【表 5】全部评测结果
"""

import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from configs.config import MODEL_CONFIG, DATA_CONFIG, OUTPUT_DIR
from baselines import (
    UnifiedLLMEngine,
    ZeroShotBaseline,
    ChatLawBaseline,
    DISCLawLLMBaseline,
    DenseRAGBaseline,
    LegalCoTBaseline,
    KGLJPBaseline,
    KGLSCRMethod
)
from evaluation.evaluator import JudicialEvaluator

def main():
    print("=" * 95)
    print("【KG-LSCR 司法智能判决预测基准评测系统 - 全算法批量复现】")
    print(f"[*] 运行模式: {MODEL_CONFIG['engine']} | 评测基座: {MODEL_CONFIG['base_model_name']}")
    print("=" * 95)

    # 1. 加载测试样本集
    cases_path = DATA_CONFIG["sample_cases_path"]
    if not os.path.exists(cases_path):
        print(f"[Error] 测试集文件不存在: {cases_path}")
        return

    with open(cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"[+] 成功加载评测样本集: {len(test_cases)} 件真实交通刑事案卷。")

    # 2. 初始化统一推理引擎
    engine = UnifiedLLMEngine(
        engine_type=MODEL_CONFIG["engine"],
        model_name=MODEL_CONFIG["base_model_name"]
    )

    # 3. 构造 7 大算法执行实例
    models = [
        ZeroShotBaseline(engine),
        ChatLawBaseline(engine),
        DISCLawLLMBaseline(engine),
        DenseRAGBaseline(engine),
        LegalCoTBaseline(engine),
        KGLJPBaseline(engine),
        KGLSCRMethod(engine)
    ]

    # 4. 执行综合评测
    evaluator = JudicialEvaluator(output_dir=OUTPUT_DIR)
    reports = []

    start_time = time.time()
    for m in models:
        rep = evaluator.evaluate_model(m, test_cases)
        reports.append(rep)

    total_time = time.time() - start_time
    print(f"\n[+] 全算法批量评测完成，总耗时: {total_time:.2f} 秒。")

    # 5. 导出统计报表与终端呈现 (匹配论文表 5)
    evaluator.save_summary_table(reports, filename="table5_benchmark_comparison.csv")

    # 同时保存 JSON 结果
    json_path = os.path.join(OUTPUT_DIR, "table5_benchmark_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"[+] JSON 格式详细报告已保存至: {json_path}")

if __name__ == "__main__":
    main()

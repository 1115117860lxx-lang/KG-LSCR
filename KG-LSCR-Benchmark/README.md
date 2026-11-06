# KG-LSCR: 基于知识图谱与法律三段式推理的交通刑事案情分析方法

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c" alt="PyTorch">
  <img src="https://img.shields.io/badge/Dataset-CAIL2018-green" alt="Dataset">
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License">
</p>

本仓库为学术论文 **《基于知识图谱与法律三段式推理的交通刑事案情分析方法》(KG-LSCR)** 的官方开源基准复现代码工程。涵盖论文中提出的 **KG-LSCR 核心算法** 以及近三年自然语言处理与智慧司法领域 **6 大主流代表性基线对比算法**（Zero-shot 直接提示、ChatLaw、DISC-LawLLM、Dense-RAG、Legal-CoT、KG-LJP）的完整评测管线、Prompt 模板、预处理代码与多维度指标计算引擎。

---

## 📌 核心研究背景与创新

交通刑事案件（交通肇事罪、危险驾驶罪）普遍存在**多主体行为交织、要件被海量冗余叙述稀释、规范法条与类案经验单库混同负迁移**以及**黑盒模型生成缺乏法理可溯性**等深层痛点。

针对上述挑战，本研究提出 **KG-LSCR** 神经-符号协同司法推理框架，涵盖四大依次承接的核心子算法：
1. **TFE (面向三态事实边界的情节抽取算法)**：显式构建十三类核心法定犯罪与量刑情节的三态规约空间 $\{-1: 未提及, 0: 明确不存在, 1: 明确存在\}$，引入主体行为归属校验算子 $\Psi_{\text{agent}}$，彻底铲除被害人违章过错张冠李戴现象。
2. **TBM-SR (局部知识图谱拓扑主干映射算法)**：构建以案件事实为核心的轻量级局部知识图谱 $G_X$，执行主干投影剪除案外人旁路分支与背景噪音，生成紧凑结构化查询 $q_{\text{KG}}$。
3. **DC-IRAG (规范-经验双通道物理隔离检索机制)**：在物理存储与内存索引层面建立严格隔离，实现权威成文法条（刚性大前提规范库）与类案判例（柔性经验库）的显式解耦召回，切断经验表述对法定定罪的跨库负迁移。
4. **LS-CoT (基于法律三段论的形式化思维链推理算法)**：重构四阶段演绎状态机（规范大前提确立 $\to$ 事实小前提提纯 $\to$ 涵摄比对与疑罪阻断排除 $\to$ 最终裁判结论与白名单校验），实现高可靠可解释司法审判辅助。

---

## 🏆 核心基准对比实验结果 (Table 5)

在 2,000 件真实 CAIL2018 交通刑事评测集与人工复核黄金评价集上的综合评测结果如下：

| 对比算法 | 模型类型与来源 | 罪名 Macro-F1 | 法条推荐 MAP | 主体归属率 (Attrib-Acc) | 事实幻觉率 (FHR) ↓ | 量刑偏差 MAE (月) ↓ | 三段论合规分 (SVS) | 规范冲突率 (NCR) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **直接提示 (Zero-shot)** | 通用开源大模型 (2024) | 75.5% | 65.8% | 68.4% | 31.2% | 13.6月 | 68.5分 | 28.4% |
| **ChatLaw** | 法律微调大模型 (2023) | 80.0% | 73.2% | 73.5% | 23.8% | 9.2月 | 74.2分 | 21.5% |
| **DISC-LawLLM** | 司法监督微调模型 (2023) | 81.5% | 75.4% | 75.2% | 21.6% | 8.5月 | 76.8分 | 18.6% |
| **Dense-RAG** | 稠密向量检索增强 (2024) | 79.7% | 74.1% | 71.8% | 25.4% | 9.8月 | 72.0分 | 24.2% |
| **Legal-CoT** | 司法思维链提示 (ICAIL 2023)| 82.5% | 77.2% | 76.9% | 19.8% | 7.6月 | 79.4分 | 16.5% |
| **KG-LJP** | 传统图谱判决预测 (2025) | 83.5% | 79.0% | 81.2% | 17.5% | 6.8月 | 81.2分 | 14.2% |
| **KG-LSCR (本文)** | **解耦图谱与三段论 (2026)** | **93.6%** | **89.2%** | **95.8%** | **3.2%** | **2.8月** | **94.6分** | **2.5%** |

---

## 🔬 逐步增量消融实验结果 (Table 6)

| 消融实验配置组合 | 罪名 Macro-F1 | 法条推荐 MAP | 主体归属率 | 事实幻觉率 ↓ | 较基线增益 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **配置一：Base 基准模型 (Qwen2.5-72B)** | 75.5% | 65.8% | 68.4% | 31.2% | 基线对照 |
| **配置二：+ TFE (三态事实边界抽取)** | 82.6% | 71.2% | 87.5% | 16.8% | +7.1% |
| **配置三：+ TBM-SR (图谱主干去噪映射)** | 86.8% | 78.5% | 91.2% | 11.5% | +11.3% |
| **配置四：+ DC-IRAG (双通道物理隔离检索)**| 90.8% | 85.6% | 93.8% | 6.4% | +15.3% |
| **配置五：+ LS-CoT (本文完整 KG-LSCR)** | **93.6%** | **89.2%** | **95.8%** | **3.2%** | **+18.1%** |

---


---

## ⚡ 检索策略与物理隔离消融实验结果 (Table 7)

| 检索策略组合 | 查询构建方式 | 索引架构策略 | 法条 Recall@1 | 法条 Recall@3 | 法条合规率 (CVR) | 规范隔离度 (CDIS) | 平均检索延迟 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **策略一 (Raw + Mixed)** | 原始千字长文叙述 | 单库混编检索 (Mixed-RAG) | 68.4% | 82.0% | 64.5% | 48.0% | 18.5ms |
| **策略二 (TBM + Mixed)** | 图谱主干查询 {	ext{KG}}$ | 单库混编检索 (Mixed-RAG) | 81.5% | 88.2% | 71.2% | 52.4% | 11.2ms |
| **策略三 (Raw + DC-IRAG)** | 原始千字长文叙述 | 双通道物理隔离 (DC-IRAG) | 83.0% | 91.5% | 88.4% | 94.5% | 14.6ms |
| **策略四 (本文完整机制)** | 图谱主干查询 {	ext{KG}}$ | 双通道物理隔离 (DC-IRAG) | **94.2%** | **98.6%** | **96.8%** | **98.2%** | **12.0ms** |

## 📂 项目工程目录结构

```text
KG-LSCR-Benchmark/
├── README.md               # 项目主说明文档
├── requirements.txt        # 环境依赖清单
├── .gitignore              # Git 忽略配置
├── configs/
│   └── config.py           # 数据路径、统一评测解码参数与模型配置
├── data/
│   ├── README.md           # 数据集说明文档
│   ├── sample_cases.json   # 50件精选平衡测试案卷（开箱即跑）
│   ├── law_library.json    # 13条交通刑事核心刑法条文库
│   ├── gold100_sample.csv  # 人工复核争议黄金集
│   └── cail2018_traffic_test_2000.json # CAIL2018 全量脱敏测试集
├── baselines/              # 7 大对比算法复现模块
│   ├── base_model.py       # 统一模型基座封装 (支持 vLLM/HF/API/Reproduction 模式)
│   ├── zero_shot.py        # 1. Zero-shot 直接提示基线
│   ├── chatlaw.py          # 2. ChatLaw 法律微调基线
│   ├── disc_lawllm.py      # 3. DISC-LawLLM 三段论微调基线
│   ├── dense_rag.py        # 4. Dense-RAG 单库混编检索基线
│   ├── legal_cot.py        # 5. Legal-CoT 司法思维链基线
│   ├── kg_ljp.py           # 6. KG-LJP 传统图嵌入基线
│   └── kg_lscr.py          # 7. KG-LSCR 本文完整方法全流程装配
├── modules/                # 本文 4 大核心子算法组件
│   ├── tri_state_extractor.py # TFE: 13类三态事实边界抽取与主体归属规约
│   ├── graph_backbone.py      # TBM-SR: 局部案件图谱构建与主干降维投影
│   ├── dual_channel_rag.py    # DC-IRAG: 法规与判例双通道物理隔离检索
│   └── syllogism_reasoner.py  # LS-CoT: 四阶段形式化法律三段论推理机
├── evaluation/             # 多维司法指标评估体系
│   ├── metrics.py          # Macro-F1, MAP, Attrib-Acc, FHR, MAE, SVS, NCR
│   └── evaluator.py        # 批量评估器与 CSV 报表生成器
├── scripts/                # 一键运行与复现脚本
│   ├── quick_demo.py       # 单案件全算法交互式推演 Demo
│   ├── run_all_baselines.py# 批量运行所有对比算法并复现表 5
│   └── run_ablation.py     # 运行逐步消融实验并复现表 6
└── outputs/                # 结果输出目录 (CSV/JSON/Log)
```

---

## 🚀 快速上手与运行指南

### 1. 环境准备
```bash
git clone https://github.com/your-username/KG-LSCR-Benchmark.git
cd KG-LSCR-Benchmark
pip install -r requirements.txt
```

### 2. 运行单案情快速推演演示 (Quick Demo)
无需连接 GPU 或网络，使用内置的轻量确定性复现引擎即可秒级验证四大算法流程：
```bash
python scripts/quick_demo.py
```

### 3. 一键批量复现全部对比算法 (Table 5)
```bash
python scripts/run_all_baselines.py
```
评测完成后，将在终端打印完整表格，并将结果导出至 `outputs/table5_benchmark_comparison.csv`。

### 4. 一键运行逐步增量消融实验 (Table 6)
```bash
python scripts/run_ablation.py
```

### 5. 切换为真实 GPU 模型推理 (可选)
如果拥有本地 GPU 算力服务器（如单机 4 卡 A100/80G 或 8 卡 RTX 4090），在 `configs/config.py` 中将 `engine` 修改为 `"vllm"`：
```python
MODEL_CONFIG = {
    "engine": "vllm",
    "base_model_name": "Qwen/Qwen2.5-72B-Instruct",  # 或 deepseek-ai/DeepSeek-R1-Distill-Qwen-70B
    "temperature": 0.1,
    "top_p": 0.9,
    "max_new_tokens": 1536
}
```
或通过环境变量配置第三方 API：
```bash
export OPENAI_API_BASE="https://api.deepseek.com/v1"
export OPENAI_API_KEY="your-api-key"
```

---

## 📜 引用说明 (Citation)

如果您在研究中使用了本代码库或实验基准，请引用我们的论文：

```bibtex
@article{long2026kglscr,
  title={基于知识图谱与法律三段式推理的交通刑事案情分析方法},
  author={龙新星 and 廖律超 and 王志民 and 胡聪沛 and 邹复民},
  journal={计算机学报 / 中文核心期刊},
  year={2026}
}
```

## 📄 许可证 (License)
本项目遵循 [Apache-2.0 License](LICENSE) 开源协议。

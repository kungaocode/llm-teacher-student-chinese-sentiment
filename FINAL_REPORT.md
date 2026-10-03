# Teacher-Student Sentiment Project: Final Close-out Report

## 结论摘要

这个项目的工程链路已经跑通：2,000 条教师标注、200 条 dev、200 条 held-out
测试、云端 4B 学生微调与部署、TF-IDF/LinearSVC 基线、成本核算和外部弱校验都有
真实产物。

当前结果适合作为 demo 和工程闭环，但不适合宣称完成“人工 gold 上的
publication-grade 性能评测”。原因是 `data/splits/gold.jsonl` 的 200 个三类
`label` 仍为空；现有主表的真值是教师建议，不是独立人工真值。

## 已完成的交付

- 数据下载、清洗、去重和固定 seed 划分
- 教师 `kimi-k3` 对训练池、dev 和 gold 候选生成三类标签
- 基于教师标签的云端 LoRA 学生微调，部署模型为
  `qwen3-4b-instruct-2507-d8e7296a858a`
- TF-IDF（char 2-4 grams）+ LinearSVC 低成本基线
- 成本统计、按类别统计、困难样本分类和 pairwise 一致性
- 使用原始二分类标签的独立弱校验脚本
- 本地 Web 人工标注台，支持中性 / 正面 / 负面确认、撤销、进度统计、
  按标签目录持久化和 A/B 双人独立标注
- 可运行的最小单元测试集

主要入口：

| 内容 | 路径 |
|---|---|
| 英文说明与主结果 | `README.md` |
| 数据来源与许可边界 | `DATA_SOURCES.md` |
| 弱二分类校验 JSON | `reports/binary_validation.json` |
| 教师建议 | `results/gold_teacher_proposals.jsonl` |
| 测试集来源标记 | `results/test_composition.jsonl` |
| 成本表 | `results/cost.json` |
| Web 标注入口 | `scripts/9_annotate_web.py` |
| 当前空人工标注模板 | `data/annotations/gold_annotation.csv` |

## 评测口径一：与教师建议的一致性

这是当前 README 主表的真实含义。教师作为自己的参照，因此教师行按构造为 1.0。

| model | accuracy | macro-F1 | neutral F1 |
|---|---:|---:|---:|
| Teacher kimi-k3 | 1.0000 | 1.0000 | 1.0000 |
| Student qwen3-4B | 0.9100 | 0.8735 | 0.7419 |
| TF-IDF + LinearSVC | 0.8050 | 0.6178 | 0.1290 |

学生与教师的三类一致率为 91.00%，基线为 80.50%，学生与基线为 79.00%。
这能说明学生成功复现了大部分教师标签，不能说明学生已经达到 91% 的人工准确率。

## 评测口径二：原始二分类外部弱校验

`source_label` 只有 0/1，而且存在噪声。`neutral` 在这里不作为预测类别：
它降低 coverage，在 neutral-as-error accuracy 中按错误计入。

| model | neutral | coverage | covered accuracy | neutral-as-error accuracy | negative recall | positive recall |
|---|---:|---:|---:|---:|---:|---:|
| Teacher kimi-k3 | 28 | 86.00% | 94.77% | 81.50% | 93.41% | 71.56% |
| Student qwen3-4B | 34 | 83.00% | 96.39% | 80.00% | 94.51% | 67.89% |
| TF-IDF + LinearSVC | 3 | 98.50% | 83.25% | 82.00% | 93.41% | 72.48% |
| Student, neutral 回退到基线 | 2 | 99.00% | 87.88% | 87.00% | 96.70% | 78.90% |

学生输出 neutral 时回退到 TF-IDF，可以把弱二分类准确率从 80.00% 提升到
87.00%。但 neutral 本身就是待人工确认的目标类别，这会牺牲三类语义，不能包装成
“三类性能提升”。它是部署时的拒绝分类/回退策略诊断。

## 成本

成本来自 `results/cost.json`，按调用时配置的 list price 估算：

| model | CNY / 1k predictions | seconds / 1k predictions |
|---|---:|---:|
| TF-IDF + LinearSVC | 0.0000 | 2.88 |
| Student qwen3-4B deployment | 0.1886 | 451.52 |
| Teacher kimi-k3 API | 1.1467 | 4459.51 |

学生相对教师约节省 6 倍 API 成本。秒数是本批次 API 墙钟时间，不是经过多次
基准测试的稳定延迟；本地基线的秒数包含拟合和推理，不能直接当作生产吞吐。

## 人工标注台与双人流程

`scripts/9_annotate_web.py` 提供零额外依赖的本地前后端标注平台。后端只监听
本机地址，前端逐条展示评论；点击中性、正面或负面按钮并确认后，标注立即写入
对应标签文件夹，同时重建兼容现有流程的 JSONL 文件。快捷键为 `1` 中性、
`2` 正面、`3` 负面。

双人独立标注建议分别启动：

```bash
# 终端 1
python3 scripts/9_annotate_web.py --annotator a --open
# 终端 2
python3 scripts/9_annotate_web.py --annotator b --port 8766 --open
```

标注文件结构：

```text
data/annotations/web/<annotator>/labels/neutral/<id>.json
data/annotations/web/<annotator>/labels/positive/<id>.json
data/annotations/web/<annotator>/labels/negative/<id>.json
data/annotations/review_<annotator>.jsonl
```

界面中的“导出标注表”只更新
`data/annotations/gold_annotation.csv` 的当前标注员列，不直接改写
`data/splits/gold.jsonl`。A/B 两人完成标注后执行：

```bash
python3 scripts/annotate_gold.py --collect
python3 scripts/3_check_gold.py
```

一致样本会被收集为三类 `label`；不一致样本会保留并列出，等待人工裁定。

## 关键限制

1. **没有独立三类 gold。** 200 条 held-out 的 `label` 全为空，教师建议不能
   同时作为标签来源和最终真值。
2. **neutral 没有外部真值。** 原始语料仅二分类，无法独立验证 28 到 34 条
   neutral 预测是否正确。
3. **源标签有噪声。** 外部弱校验只能提供辅助信号，不能替代双人标注和仲裁。
4. **配置与部署不完全同名。** `configs/default.yaml` 保留本地
   `Qwen/Qwen3-0.6B` 参考配置；正式结果来自 Bailian 上的 4B 部署。
5. **许可未明确。** 两个上游数据分发都没有明确 licence 字段，研究使用备注
   不等于许可授权。
6. **教师上限是构造结果。** 教师行 1.000 不代表人工准确率，只表示它在当前
   教师建议真值上完全一致。

## 复现与验证

```bash
python3 scripts/8_validate_binary.py
python3 -m pytest -q
python3 scripts/4_train_student.py --dry-run
```

当前验证结果：

- `scripts/8_validate_binary.py` 在 200 条 gold id 上完成对齐并写出 JSON/Markdown
- `pytest`：11 passed
- `4_train_student.py --dry-run`：train 2000、dev 200，标签分布和 balanced
  weights 可计算

## 收尾判断

工程 demo：完成。

论文/CV 的严格数字：尚缺独立人工三类 gold 和明确数据许可。下一步只需完成 200
条双人标注与仲裁，把 `data/splits/gold.jsonl` 的 `label` 补齐，再重跑
`export_finetune.py`、`score_test.py` 和 `8_validate_binary.py`；无需重新训练
学生模型。

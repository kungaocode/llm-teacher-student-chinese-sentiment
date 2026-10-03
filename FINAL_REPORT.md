# Teacher-Student Sentiment Project: Final Close-out Report

## 结论摘要

工程链路已完整跑通：2,000 条教师标注、200 条 dev、200 条 held-out
测试、云端 4B 学生微调与部署、TF-IDF/LinearSVC 基线、成本核算、人工标注台和
外部弱校验都有真实产物。

本次最终评测使用 200 条人工标签作为 gold。网页端按 A/B 双人独立标注设计，
支持双列导出、一致性检查和分歧仲裁；本次收尾直接采用 B 列结果，A 列未计入
最终标签，因此报告不把它表述为已经完成的双人仲裁评测。

主结果：教师 `kimi-k3` 为 accuracy 0.8350 / macro-F1 0.7638，学生
`qwen3-4b-instruct-2507` 为 0.8350 / 0.7709，TF-IDF + LinearSVC 为
0.7600 / 0.5654。学生与教师逐条一致率为 91.00%，说明蒸馏保留了大部分教师
行为；该一致率不是人类准确率。

## 已完成的交付

- 数据下载、清洗、去重和固定 seed 划分
- 教师 `kimi-k3` 对训练池、dev 和 gold 候选生成三类标签
- 基于教师标签的云端 LoRA 学生微调，部署模型为
  `qwen3-4b-instruct-2507-d8e7296a858a`
- TF-IDF（char 2-4 grams）+ LinearSVC 低成本基线
- 200 条人工 gold；分布为正面 91、负面 84、中性 25
- 网页端人工标注台：A/B 身份、提示确认、按标签目录持久化、撤销、导出 CSV
- 教师、学生、基线在统一人工 gold 上的离线评测，不重复调用教师 API
- 成本统计、困难样本分类、逐类别结果和原始二分类外部弱校验
- 可运行的最小单元测试集

主要入口：

| 内容 | 路径 |
|---|---|
| 英文说明与主结果 | `README.md` |
| 网页标注思路与运行说明 | `ANNOTATION_WEB.md` |
| 数据来源与许可边界 | `DATA_SOURCES.md` |
| 人工标注 CSV（最终使用 B 列） | `data/annotations/gold_annotation.csv` |
| 人工三类 gold | `data/splits/gold.jsonl` |
| 统一评测结果 | `results/metrics.json` |
| 外部弱校验 | `reports/binary_validation.json` |
| 测试集来源标记 | `results/test_composition.jsonl` |
| 成本表 | `results/cost.json` |

## 最终评测：人工 gold

Gold 为 B 列的 200 条人工标签，`finetune_test.jsonl` 的 truth-source 全部为
`human`。以下指标均由 `scripts/6_evaluate.py` 在同一组 200 条 held-out 行上计算。

| model | accuracy | macro-F1 | neutral F1 | kappa vs. human |
|---|---:|---:|---:|---:|
| Teacher kimi-k3 | 0.8350 | 0.7638 | 0.5283 | 0.7283 |
| **Student qwen3-4B** | **0.8350** | **0.7709** | **0.5424** | **0.7327** |
| TF-IDF + LinearSVC | 0.7600 | 0.5654 | 0.0714 | 0.5787 |

学生与教师的 accuracy 相同，学生 macro-F1 高 0.0071。由于测试集只有 200 条，
这个差距不足以宣称学生普遍优于教师；更稳妥的结论是学生在这组样本上追平了教师，
同时显著优于传统基线。

逐类别 pairwise 一致性：

| pair | agreement | Cohen's kappa |
|---|---:|---:|
| Student vs. Teacher | 0.9100 | 0.8540 |
| Teacher vs. Baseline | 0.8050 | 0.6579 |
| Student vs. Baseline | 0.7900 | 0.6393 |

学生分类别 accuracy：

| category | correct / total | accuracy |
|---|---:|---:|
| plain | 117 / 132 | 0.886 |
| sarcasm-candidate | 12 / 14 | 0.857 |
| colloquial | 6 / 8 | 0.750 |
| neutral-candidate | 42 / 60 | 0.700 |
| implicit negation | 35 / 50 | 0.700 |
| mixed positive/negative | 33 / 48 | 0.688 |

## 原始二分类外部弱校验

`source_label` 只有 0/1，而且存在噪声。`neutral` 在这里不作为预测类别：
它降低 coverage，在 neutral-as-error accuracy 中按错误计入。

| model | neutral | coverage | covered accuracy | neutral-as-error accuracy | negative recall | positive recall |
|---|---:|---:|---:|---:|---:|---:|
| Teacher kimi-k3 | 28 | 86.00% | 94.77% | 81.50% | 93.41% | 71.56% |
| Student qwen3-4B | 34 | 83.00% | 96.39% | 80.00% | 94.51% | 67.89% |
| TF-IDF + LinearSVC | 3 | 98.50% | 83.25% | 82.00% | 93.41% | 72.48% |
| Student, neutral 回退到基线 | 2 | 99.00% | 87.88% | 87.00% | 96.70% | 78.90% |

学生输出 neutral 时回退到 TF-IDF，可以把弱二分类准确率从 80.00% 提升到
87.00%。但 neutral 本身就是目标类别，这会牺牲三类语义，不能包装成
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

## 网页端人工标注工作流

网页端的完整设计、数据流、API 和复现命令见
[`ANNOTATION_WEB.md`](ANNOTATION_WEB.md)。核心流程是：

```text
gold.jsonl
  -> Web 标注台（A/B 独立身份）
  -> data/annotations/web/<annotator>/labels/<label>/<id>.json
  -> 重建 annotations.jsonl / review_<annotator>.jsonl
  -> 导出 gold_annotation.csv
  -> annotate_gold.py --collect
  -> export_finetune.py
  -> 6_evaluate.py
```

平台默认按双人流程工作：

```bash
# 终端 1
python3 scripts/9_annotate_web.py --annotator a --open
# 终端 2
python3 scripts/9_annotate_web.py --annotator b --port 8766 --open
```

两人完成后执行：

```bash
python3 scripts/annotate_gold.py --collect
```

本次收尾实际只有 B 列：

```bash
python3 scripts/annotate_gold.py --collect --allow-single-annotator
```

单人模式仍会写回人工标签，但不会虚构 Cohen's kappa，也不能描述为双人仲裁结果。

## 关键限制

1. **本次最终 gold 来自单个标注者 B。** 网页端支持 A/B 双人，但当前交付没有
   第二份独立标签，因此不能报告标注者间一致性或仲裁可靠性。
2. **neutral 样本较少。** 200 条里只有 25 条人工 neutral，其 F1 为
   0.5424，容易受少数边界样本影响。
3. **源标签有噪声。** 外部弱校验只能提供辅助信号，不能替代人工三类 gold。
4. **配置与部署不完全同名。** `configs/default.yaml` 保留本地
   `Qwen/Qwen3-0.6B` 参考配置；正式结果来自 Bailian 上的 4B 部署。
5. **许可未明确。** 两个上游数据分发都没有明确 licence 字段，研究使用备注
   不等于许可授权。
6. **测试集规模有限。** 200 条适合工程收尾，但逐类别和模型间小差距仍应谨慎解释。

## 复现与验证

已完成的最终链路：

```bash
python3 scripts/annotate_gold.py --collect --allow-single-annotator
python3 scripts/export_finetune.py
python3 scripts/score_test.py results/predictions/qwen3-4b-instruct-2507-d8e7296a858a_predictions.jsonl
python3 scripts/score_test.py results/baseline_predictions.jsonl
python3 scripts/6_evaluate.py
python3 scripts/8_validate_binary.py
```

仓库级验证：

```bash
python3 -m pytest -q
python3 -m compileall -q annotator scripts tests src
python3 scripts/6_evaluate.py
git diff --check
```

`6_evaluate.py` 复用已有的
`results/gold_teacher_proposals.jsonl` 对教师离线评分，不会再次调用教师 API；
学生使用既有 predictions，不需要重新训练。

## 收尾判断

工程 demo、人工标注平台和单人人工 gold 评测均已完成。对于当前个人收尾目标，
这组结果足以形成可复现的 README、最终报告、网页标注说明和主评测表。

如果后续需要论文/CV 级别的双人标注可靠性，只需补 A 列到现有 CSV，运行
`annotate_gold.py --collect` 处理一致项并对分歧做仲裁，再重跑
`export_finetune.py` 和 `6_evaluate.py`。这不会要求重新训练学生模型。

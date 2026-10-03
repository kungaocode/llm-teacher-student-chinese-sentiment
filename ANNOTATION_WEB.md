# 网页端人工标注工作流

这个页面用于人工检查 held-out 评论并生成三类情感 gold。目标不是做一个通用标注
SaaS，而是把“逐条阅读、确认标签、立即落盘、可恢复、可导出、可直接进入评测”
这条链路做完整。

## 设计原则

- **本地优先。** 服务默认只监听 `127.0.0.1`，数据不上传第三方。
- **零额外依赖。** 后端使用 Python 标准库 `http.server`，前端使用原生
  HTML/CSS/JavaScript，不要求 Node 或数据库。
- **确认后落盘。** 点击标签后先弹出确认对话框；确认后立即写入磁盘，避免一次
  批量保存造成整批丢失。
- **文件是事实来源。** 每条标注保存为独立 JSON；JSONL 和 CSV 都是可重建的
  索引/导出视图。
- **A/B 可独立。** 标注员身份决定写入哪一列，两个浏览器会话不会互相覆盖。
- **可直接接评测。** 收集后的标签写入 `data/splits/gold.jsonl`，再由现有
  `export_finetune.py` 和 `6_evaluate.py` 使用。

## 使用方式

单人收尾：

```bash
python3 scripts/9_annotate_web.py --annotator b --open
```

双人独立标注：

```bash
# 终端 1
python3 scripts/9_annotate_web.py --annotator a --open

# 终端 2
python3 scripts/9_annotate_web.py --annotator b --port 8766 --open
```

浏览器默认打开 `http://127.0.0.1:8765`。快捷键：

- `1`：中立
- `2`：正面
- `3`：负面
- 左右方向键：上一条 / 下一条

每次选择会弹出确认框；确认后标签写入对应类别目录。页面支持：

- 未标注 / 全部队列筛选
- 当前批次进度和标签分布
- 最近标注回看
- 撤销单条标注
- 导出当前标注员列到 `gold_annotation.csv`

## 数据流

```text
data/splits/gold.jsonl
        |
        v
scripts/9_annotate_web.py
        |
        v
data/annotations/web/<annotator>/labels/<label>/<id>.json
        |
        +--> data/annotations/web/<annotator>/annotations.jsonl
        +--> data/annotations/review_<annotator>.jsonl
        +--> data/annotations/gold_annotation.csv
        |
        v
scripts/annotate_gold.py --collect
        |
        v
data/splits/gold.jsonl (human labels)
        |
        v
scripts/export_finetune.py -> scripts/6_evaluate.py
```

`labels/<label>/<id>.json` 是源文件；页面每次变更都会重建 JSONL 索引，所以即使
服务中断，重新启动后仍能恢复当前进度。

## A/B 收集规则

平台支持三类结果：

1. A/B 都有且标签一致：直接作为最终标签。
2. A/B 都有但标签不一致：保留分歧，等待在 `final_label` 中填写人工仲裁结果。
3. 只有 A 或只有 B：这是单标注者模式，必须显式确认：

```bash
python3 scripts/annotate_gold.py --collect --allow-single-annotator
```

单人模式不会生成标注者间 Cohen's kappa。代码会明确输出这一点，报告也不能把它
描述为已经完成双人仲裁。

如果做正式双人评测，先分别完成 A/B，然后执行：

```bash
python3 scripts/annotate_gold.py --collect
```

对于有分歧的行，在 CSV 的 `final_label` 填入讨论后的定论，再重新运行
`--collect`。

## 存储布局

```text
data/annotations/
├── gold_annotation.csv
├── review_<annotator>.jsonl
└── web/
    └── <annotator>/
        ├── annotations.jsonl
        └── labels/
            ├── negative/<id>.json
            ├── neutral/<id>.json
            └── positive/<id>.json
```

Web 标注目录和大体积数据默认被 `.gitignore` 忽略，仓库只保留可复现的代码和
模板；如果要发布人工标签，应另外确认数据许可和隐私边界。

## API

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/session` | 读取 gold 行、当前标注员标签和进度 |
| `POST` | `/api/annotations` | 确认写入 `{id, label}` |
| `DELETE` | `/api/annotations/<id>` | 撤销单条标注 |
| `POST` | `/api/export/csv` | 导出当前标注员列到 CSV |

## 为什么用文件而不是数据库

这个项目的瓶颈是人工判读，不是并发写入。文件方案让每个标签都可读、可 diff、
可直接进入 Git 忽略目录，也能在服务崩溃后逐条恢复。SQLite 只有在多人远程协作、
高并发或复杂查询成为真实需求时才值得引入。

## 评测闭环

```bash
python3 scripts/annotate_gold.py --collect --allow-single-annotator
python3 scripts/export_finetune.py
python3 scripts/6_evaluate.py
```

`6_evaluate.py` 复用已有 teacher proposals 对教师离线评分，不会再调用教师 API；
学生复用已部署模型的 predictions，不需要重新训练。

## 验证

相关单元测试覆盖标注落盘、标签迁移、撤销、CSV 单列更新和非法 ID 拒绝：

```bash
python3 -m pytest -q tests/test_annotation_store.py
```

完整验证：

```bash
python3 -m pytest -q
python3 -m compileall -q annotator scripts tests src
```

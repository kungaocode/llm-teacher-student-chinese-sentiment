"""Heuristic for flagging candidate-neutral reviews (plan §6, §4.2).

The public corpora are 2-class (pos/neg); ``neutral`` has no ground truth. This
module scores reviews that are *plausibly* neutral — an explicit neutral/mixed
marker, or both positive and negative sentiment words in one review — so the
gold split can reserve slots for them. The teacher + human double-review then
set the real 3-class label.

The score only steers sampling; it never decides a label. It is deliberately
loose (false positives are expected), so ``neutral_score`` returns a rank, not a
classification.
"""
from __future__ import annotations

# Explicit neutral / mixed markers. Multi-char to limit substring noise.
NEUTRAL_HINTS: tuple[str, ...] = (
    "一般般", "中规中矩", "不好不坏", "马马虎虎", "无功无过", "喜忧参半",
    "有好有坏", "好坏参半", "说不上好坏", "不温不火", "说不上", "谈不上",
    "没感觉", "没啥感觉", "就那样", "也就那样", "差不多", "过得去", "还凑合",
    "凑合", "尚可", "及格", "平平", "中等", "普通", "中评", "一般", "还行",
    "还可以", "还好", "不太好", "不太行", "不怎么", "没那么好", "勉强",
    "性价比一般",
)

POS_HINTS: tuple[str, ...] = (
    "物美价廉", "性价比高", "五星好评", "赞不绝口", "爱了", "给力", "超值",
    "完美", "惊喜", "点赞", "很棒", "优秀", "满意", "推荐", "热情", "干净",
    "舒服", "划算", "美味", "便捷", "贴心", "实惠", "不错", "漂亮", "喜欢",
    "好评", "五星", "棒", "赞", "值", "好",
)

NEG_HINTS: tuple[str, ...] = (
    "失望透顶", "上当", "受骗", "不好用", "难用", "体验差", "不推荐", "假货",
    "劣质", "坑爹", "差劲", "恶心", "生气", "愤怒", "不值", "失望", "糟糕",
    "异味", "难闻", "退货", "差评", "问题", "太贵", "垃圾", "不行", "后悔",
    "坑", "脏", "吵", "慢", "坏", "差", "烂",
)


def hint_counts(text: str) -> tuple[int, int, int]:
    """Return ``(n_neutral, n_positive, n_negative)`` *distinct* hint hits.

    Counting distinct words (not occurrences) stops spammy repeated-word reviews
    (``一般一般一般``) from inflating the score and crowding out genuine neutral
    reviews.
    """
    return (
        sum(1 for w in NEUTRAL_HINTS if w in text),
        sum(1 for w in POS_HINTS if w in text),
        sum(1 for w in NEG_HINTS if w in text),
    )


def neutral_score(text: str) -> int:
    """0 = not a neutral candidate; higher = more likely neutral.

    Explicit neutral markers count double (strong signal); a review that mixes
    positive and negative words counts once (hedged / mixed). The gold split
    takes the top-scoring candidates.
    """
    n_neu, n_pos, n_neg = hint_counts(text)
    if n_neu == 0 and not (n_pos and n_neg):
        return 0
    return n_neu * 2 + (1 if (n_pos and n_neg) else 0)


def matched_hints(text: str) -> dict[str, list[str]]:
    """Return the specific hint words present in ``text`` (for human inspection)."""
    return {
        "neutral": [w for w in NEUTRAL_HINTS if w in text],
        "positive": [w for w in POS_HINTS if w in text],
        "negative": [w for w in NEG_HINTS if w in text],
    }

"""Teacher LLM interface.

The teacher is a *labeler*, never a trained model: any strong OpenAI-compatible
chat endpoint works (DashScope / DeepSeek / Moonshot / Zhipu / generic OpenAI).
Provider presets (base_url / model / api-key env var) live in ``PROVIDERS`` and
are overridable per-run via config ``teacher.base_url`` / ``teacher.model`` /
``teacher.api_key_env``. A ``mock`` provider (offline rule-based polarity) is
available for smoke tests with no API key.

Each ``label(text)`` returns ``(label, reason, UsageRecord)`` so the caller can
accumulate token/latency usage for the cost step.
"""
from __future__ import annotations

import os
import time

from .config import Config
from .cost import UsageRecord
from .labels import parse_teacher_json

# Strip any socks:// proxy (e.g. Clash on 127.0.0.1:7897) before httpx is used —
# httpx cannot parse the socks scheme and would crash on client construction.
# Mirrors the same guard at the top of scripts/0_download.py.
for _proxy_key in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(_proxy_key, None)

# Few-shot examples (text, label, reason).
EXAMPLES = [
    ("酒店位置很好，服务热情，房间干净，下次还会住。", "positive", "通篇好评"),
    ("价格实惠，东西也好用，非常满意。", "positive", "明确正面评价"),
    ("整体一般般，位置还行，其他都普通。", "neutral", "无强烈褒贬"),
    ("这家店中规中矩，不好也不坏。", "neutral", "中性评价"),
    ("房间有异味，隔音很差，不推荐入住。", "negative", "明确负面"),
    ("服务太差，等了很久，很失望。", "negative", "明确负面"),
]

_SYS = (
    "你是一名中文评论情感标注员。请把评论分为三类之一：positive / neutral / negative。"
    "只输出一个 JSON 对象，格式 {\"label\": \"...\", \"reason\": \"...\"}，"
    "reason 用一句话简短说明依据。不要输出其它文字。"
)

# Provider presets: default base_url / model / env var holding the API key.
# Model names are the latest-known per provider — confirm against each provider's
# current docs before final cost reporting (config overrides these anyway).
PROVIDERS: dict[str, dict] = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_env": "DASHSCOPE_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "key_env": "MOONSHOT_API_KEY",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
        "key_env": "ZHIPU_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
}


def build_messages(text: str, examples: list[tuple[str, str, str]] = EXAMPLES) -> list[dict]:
    msgs = [{"role": "system", "content": _SYS}]
    for ex_text, ex_label, ex_reason in examples:
        msgs.append({"role": "user", "content": ex_text})
        msgs.append(
            {"role": "assistant",
             "content": '{"label": "%s", "reason": "%s"}' % (ex_label, ex_reason)}
        )
    msgs.append({"role": "user", "content": text})
    return msgs


class MockTeacher:
    """Rule-based polarity for offline smoke tests. Not a real teacher."""

    model = "mock"
    _NEG = ["差", "烂", "垃圾", "异味", "难闻", "脏", "吵", "慢", "坑", "失望",
            "不推荐", "后悔", "糟糕", "恶劣", "陈旧", "隔音差", "太贵", "不行"]
    _POS = ["好", "棒", "赞", "满意", "推荐", "热情", "干净", "舒服", "划算",
            "值", "美味", "便捷", "贴心", "优秀", "不错", "漂亮", "实惠"]

    def label(self, text: str) -> tuple[str, str, UsageRecord]:
        pos = sum(text.count(w) for w in self._POS)
        neg = sum(text.count(w) for w in self._NEG)
        if neg > pos:
            label, reason = "negative", "负面词多于正面词"
        elif pos > neg:
            label, reason = "positive", "正面词多于负面词"
        else:
            label, reason = "neutral", "无显著倾向"
        return label, reason, UsageRecord(model=self.model, n_predictions=1)


class OpenAICompatTeacher:
    """Any OpenAI-compatible chat endpoint, selected by ``teacher.provider``."""

    def __init__(self, cfg: Config):
        provider = cfg.get("teacher.provider", "dashscope")
        preset = PROVIDERS.get(provider)
        if preset is None:
            raise ValueError(
                f"unknown teacher.provider {provider!r}; known: {sorted(PROVIDERS)}"
            )

        self.base_url = cfg.get("teacher.base_url") or preset["base_url"]
        self.model = cfg.get("teacher.model") or preset["model"]
        key_env = cfg.get("teacher.api_key_env") or preset["key_env"]

        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"{key_env} is not set (see .env). provider={provider} expects {key_env}."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("install the 'openai' package (pip install openai)") from exc

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        _temp = cfg.get("teacher.temperature", 0.0)
        self.temperature = None if _temp is None else float(_temp)
        self.max_tokens = int(cfg.get("teacher.max_tokens", 256))
        self.max_retries = int(cfg.get("teacher.max_retries", 3))
        # Optional knob for reasoning models (K3). None = don't send the field.
        self.reasoning_effort = cfg.get("teacher.reasoning_effort")

    def label(self, text: str) -> tuple[str, str, UsageRecord]:
        messages = build_messages(text)
        last_exc = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                }
                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""
                usage = resp.usage
                label, reason = parse_teacher_json(content)
                rec = UsageRecord(
                    model=self.model,
                    n_predictions=1,
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    seconds=time.time() - t0,
                )
                return label, reason, rec
            except Exception as exc:  # noqa: BLE001 - retry on API/parse errors
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"teacher failed after {self.max_retries} retries: {last_exc}")


def make_teacher(cfg: Config):
    provider = cfg.get("teacher.provider", "mock")
    if provider == "mock":
        return MockTeacher()
    return OpenAICompatTeacher(cfg)

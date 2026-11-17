"""Conservative lexical matching; scores are heuristics, not probabilities."""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

import jieba

STOP_WORDS = set(
    "的 了 呢 吗 啊 是 在 中 里 与 和 及 一个 一种 通过 使用 进行 应该 需要 如何 怎么 怎样 什么 为什么 请 请问 请教 解释 一下 含义 理解 意思 不同 区别 方法 步骤 原因 关于 有 有什么 时候 时 下 后 上 个 我 你 老师 您好 你好 可以 该 能够 应当".split()
)
GENERIC_TERMS = set(
    "问题 错误 报错 失败 解决 处理 调试 出现 发生 情况 东西 内容 安装 配置 设置 更新 恢复 运行 代码 编程".split()
)
SYNONYMS = {
    "转化": "转换",
    "互相": "相互",
    "出错": "报错",
    "不成功": "失败",
    "不精确": "不精准",
    "大整数": "大数字",
    "ubantu": "ubuntu",
}
CONTEXT_LANGUAGES = {"python", "java"}
ACTION_PAIRS = [
    ("安装", "卸载"),
    ("导入", "导出"),
    ("开启", "关闭"),
    ("启用", "禁用"),
    ("增加", "减少"),
]


def normalized(text):
    # Preserve symbols that distinguish regexes, comparison operators and identifiers.
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\s，。！？、;；？!,：]+", "", text)


def tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    for old, new in SYNONYMS.items():
        text = text.replace(old, new)
    parts = []
    # Technical identifiers, versions and error codes remain whole tokens.
    for fragment in re.findall(
        r"[a-z_][a-z0-9_.-]*|\d+(?:\.\d+)*|[\u4e00-\u9fff]+|[\^$!=<>|]+", text
    ):
        words = jieba.lcut(fragment) if re.fullmatch(r"[\u4e00-\u9fff]+", fragment) else [fragment]
        parts.extend(word for word in words if word and word not in STOP_WORDS)
    result = set(parts)
    if any(re.search(r"[a-z]", part) for part in result):
        result -= {"系统", "编辑器", "软件", "工具"}
    return result


def technical_anchors(text):
    anchors = set(re.findall(r"[a-z_][a-z0-9_.-]*", unicodedata.normalize("NFKC", text).casefold()))
    anchors = {SYNONYMS.get(a, a) for a in anchors}
    return anchors - CONTEXT_LANGUAGES


def incompatible(query, candidate):
    query_anchors = technical_anchors(query)
    candidate_anchors = technical_anchors(candidate)
    if query_anchors and not query_anchors.issubset(candidate_anchors):
        return "technical_anchor_mismatch"
    q_numbers = set(re.findall(r"(?<![a-z])\d+(?:\.\d+)*(?![a-z])", query.casefold()))
    c_numbers = set(re.findall(r"(?<![a-z])\d+(?:\.\d+)*(?![a-z])", candidate.casefold()))
    if q_numbers and c_numbers and not q_numbers.issubset(c_numbers):
        return "number_or_version_mismatch"
    for positive, opposite in ACTION_PAIRS:
        if (positive in query and opposite in candidate) or (
            opposite in query and positive in candidate
        ):
            return "opposite_action"
    exclusions = r"不使用|不用|不依赖|不借助|不要|无需|禁止"
    if re.search(exclusions, query) and not re.search(exclusions, candidate):
        return "explicit_exclusion"
    q_operators = set(re.findall(r"\^|\$|!=|==|<=|>=", query))
    c_operators = set(re.findall(r"\^|\$|!=|==|<=|>=", candidate))
    if q_operators and not q_operators.issubset(c_operators):
        return "operator_mismatch"
    return None


@dataclass(frozen=True)
class Match:
    question: str
    score: float
    coverage: float
    reason: str
    raw_score: float = 0.0


class LexicalMatcher:
    def __init__(self, questions):
        frequency = Counter(term for question in questions for term in tokens(question))
        count = len(questions)
        self.idf = {term: math.log(1 + count / (1 + frequency[term])) for term in frequency}

    def compare(self, query, candidate):
        q_normal, c_normal = normalized(query), normalized(candidate)
        if q_normal and q_normal == c_normal:
            return Match(candidate, 1.0, 1.0, "exact")
        blocked = incompatible(query, candidate)
        if blocked:
            return Match(candidate, 0.0, 0.0, blocked)
        query_terms, candidate_terms = tokens(query), tokens(candidate)
        if not query_terms - GENERIC_TERMS - CONTEXT_LANGUAGES:
            return Match(candidate, 0.0, 0.0, "insufficient_topic")
        shared = query_terms & candidate_terms

        def weight(terms):
            return sum(self.idf.get(t, 1.0) for t in terms)

        coverage = weight(shared) / weight(query_terms) if query_terms else 0.0
        precision = weight(shared) / weight(candidate_terms) if candidate_terms else 0.0
        f1 = 2 * coverage * precision / (coverage + precision) if coverage + precision else 0.0
        clean_query = "".join(sorted(query_terms))
        clean_candidate = "".join(sorted(candidate_terms))
        similarity = SequenceMatcher(None, clean_query, clean_candidate).ratio()
        score = max(f1, similarity * 0.9) if coverage >= 0.72 else 0.0
        return Match(candidate, score, coverage, "lexical")

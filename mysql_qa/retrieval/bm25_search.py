"""BM25 candidate recall plus conservative lexical acceptance and scoped caching."""

import hashlib
import time
from dataclasses import dataclass, replace

from rank_bm25 import BM25Okapi

from base import logger

from .answer_quality import is_usable_answer
from .lexical_match import LexicalMatcher, Match, normalized, tokens


@dataclass(frozen=True)
class FAQCandidate:
    question: str
    answer: str
    source: str
    lexical_score: float
    raw_score: float


class BM25Search:
    POLICY_VERSION = "v3.2-body-quality"

    def __init__(self, redis_client, mysql_client, *, clock=time.monotonic):
        self.logger = logger
        self.redis_client = redis_client
        self.mysql_client = mysql_client
        self.clock = clock
        self.indexes = {}
        self._index(None)

    @staticmethod
    def _normalize_source_filter(source_filter):
        if not source_filter:
            return None
        value = str(source_filter).strip()
        # 'java' is a separate document scope, so do not map it to the QA subject.
        return {"python": "Python学科", "python学科": "Python学科", "java学科": "JAVA学科"}.get(
            value.lower(), value
        )

    def refresh(self):
        self.indexes.clear()

    def _index(self, source_filter):
        cached = self.indexes.get(source_filter)
        if cached and self.clock() - cached["created"] < 60:
            return cached
        rows = (
            self.mysql_client.fetch_questions(source_filter)
            if source_filter
            else self.mysql_client.fetch_questions()
        )
        questions = list(
            dict.fromkeys(str(row[0] if isinstance(row, (tuple, list)) else row) for row in rows)
        )
        documents = [
            (question, alternative.strip())
            for question in questions
            for alternative in question.splitlines()
            if alternative.strip()
        ]
        term_lists = [
            list(tokens(alternative)) or [normalized(alternative)] for _, alternative in documents
        ]
        fingerprint = hashlib.sha256("\n".join(questions).encode("utf-8")).hexdigest()[:16]
        index = {
            "created": self.clock(),
            "questions": questions,
            "documents": documents,
            "bm25": BM25Okapi(term_lists) if term_lists else None,
            "matcher": LexicalMatcher([alternative for _, alternative in documents]),
            "fingerprint": fingerprint,
        }
        self.indexes[source_filter] = index
        return index

    def rank(self, query, source_filter=None):
        source_filter = self._normalize_source_filter(source_filter)
        index = self._index(source_filter)
        if not index["documents"]:
            return []
        scores = index["bm25"].get_scores(list(tokens(query)))
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:12]
        # Exact aliases must still work when corpus-wide BM25 IDF is zero/negative.
        order.extend(
            i
            for i, (_, alias) in enumerate(index["documents"])
            if normalized(alias) == normalized(query)
        )
        unique = {}
        # A stored row may contain several aliases; the complete row is also a valid exact query.
        for question in index["questions"]:
            if normalized(question) == normalized(query):
                unique[question] = Match(question, 1.0, 1.0, "exact")
        for position in dict.fromkeys(order):
            question, alias = index["documents"][position]
            match = index["matcher"].compare(query, alias)
            match = replace(match, question=question, raw_score=float(scores[position]))
            if question not in unique or match.score > unique[question].score:
                unique[question] = match
        return sorted(
            unique.values(), key=lambda m: (m.score, m.coverage, m.raw_score), reverse=True
        )

    def search(self, query, threshold=0.85, source_filter=None):
        if not isinstance(query, str) or not query.strip():
            return None, True
        source_filter = self._normalize_source_filter(source_filter)
        try:
            index = self._index(source_filter)
            digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
            cache_key = f"answer:{self.POLICY_VERSION}:{index['fingerprint']}:{source_filter or 'all'}:{threshold}:{digest}"
            cached = self.redis_client.get_data(cache_key)
            if is_usable_answer(cached):
                return cached, False
            ranked = self.rank(query, source_filter)
            if not ranked:
                return None, True
            best = ranked[0]
            # A score is not a calibrated confidence probability. Ambiguous candidates abstain.
            ambiguous = (
                len(ranked) > 1 and best.reason != "exact" and best.score - ranked[1].score < 0.06
            )
            if best.score < threshold or ambiguous:
                return None, True
            answer = self.mysql_client.fetch_answer(best.question, source_filter)
            if is_usable_answer(answer):
                self.redis_client.set_data(cache_key, answer)
                return answer, False
        except Exception as exc:
            self.logger.error("标准问答检索失败：%s", type(exc).__name__)
        return None, True

    def recall_candidates(self, query, source_filter, *, limit=12):
        """Recall scoped FAQ bodies; this does not authorize returning an answer."""
        scope = self._normalize_source_filter(source_filter)
        if not scope or not isinstance(query, str) or not query.strip():
            return []
        candidates = []
        for match in self.rank(query, scope)[: max(0, min(limit, 12))]:
            answer = self.mysql_client.fetch_answer(match.question, scope)
            if is_usable_answer(answer):
                candidates.append(
                    FAQCandidate(match.question, answer, scope, match.score, match.raw_score)
                )
        return candidates

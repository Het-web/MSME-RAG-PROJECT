import json
import re
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from llm import get_sufficiency_llm
from prompts import CONTEXT_SUFFICIENCY_PROMPT
from utils import elapsed_ms, now_ms


MIN_CONTEXT_CHARS = 300

TEMPORAL_QUERY_PATTERN = re.compile(
    r"\b(future|upcoming|latest|recent|new|announced?|notifications?|updates?|changes?|circulars?|guidelines?|policy update)\b",
    re.IGNORECASE,
)

MSME_DEFINITION_QUERY_PATTERN = re.compile(
    r"\b(definition|define|classification|classify|what is an? msme|what are msmes?)\b",
    re.IGNORECASE,
)

MSME_THRESHOLD_CONTEXT_PATTERNS = [
    re.compile(r"\bmicro\b", re.IGNORECASE),
    re.compile(r"\bsmall\b", re.IGNORECASE),
    re.compile(r"\bmedium\b", re.IGNORECASE),
    re.compile(r"\binvestment\b", re.IGNORECASE),
    re.compile(r"\bturnover\b", re.IGNORECASE),
    re.compile(r"\b(?:rs\.?|inr|rupees?|crores?|lakh|lakhs?|[0-9])\b", re.IGNORECASE),
]

DETAIL_INTENT_PATTERNS = {
    "eligibility criteria": re.compile(r"\b(eligib(?:le|ility)|criteria|qualif(?:y|ication))\b", re.IGNORECASE),
    "benefits": re.compile(r"\b(benefits?|advantages?|support provided|assistance)\b", re.IGNORECASE),
    "application process": re.compile(r"\b(apply|application|process|procedure|how to apply)\b", re.IGNORECASE),
    "loan limits": re.compile(r"\b(loan limit|credit limit|maximum loan|loan amount|coverage amount)\b", re.IGNORECASE),
    "subsidy amounts": re.compile(r"\b(subsidy|subsidies|grant amount|assistance amount)\b", re.IGNORECASE),
    "scheme details": re.compile(r"\b(scheme details?|details?|features?|components?)\b", re.IGNORECASE),
    "registration requirements": re.compile(r"\b(registration requirements?|register|udyam|documents? required)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class SufficiencyResult:
    answerable: bool
    reason: str
    metrics: dict[str, bool | float | str]


class ContextSufficiencyEvaluator:
    def __init__(self) -> None:
        self.prompt = ChatPromptTemplate.from_template(CONTEXT_SUFFICIENCY_PROMPT)

    def evaluate(self, query: str, context: str, chunk_count: int) -> SufficiencyResult:
        start = now_ms()
        context = context.strip()

        if chunk_count == 0:
            return self._result(False, "No retrieval results exist.", start)

        if not context:
            return self._result(False, "Retrieved context is empty.", start)

        conservative_reason = self._conservative_missing_reason(query, context)
        if conservative_reason:
            return self._result(False, conservative_reason, start)

        if len(context) < MIN_CONTEXT_CHARS:
            return self._result(False, "Retrieved context is too short to fully answer the question.", start)

        response = (self.prompt | get_sufficiency_llm()).invoke({"query": query, "context": context})
        answerable, llm_reason = self._parse_evaluation(str(response.content))
        if answerable:
            return self._result(True, llm_reason or "Retrieved context is sufficient.", start)

        missing_detail = self._missing_detail_reason(query, context)
        reason = missing_detail or llm_reason or "Retrieved context does not fully answer the question."
        return self._result(False, reason, start)

    def _result(self, answerable: bool, reason: str, start: float) -> SufficiencyResult:
        return SufficiencyResult(
            answerable=answerable,
            reason=reason,
            metrics={
                "context_sufficiency_check_ms": elapsed_ms(start),
                "context_answerable": answerable,
                "context_sufficiency_reason": reason,
            },
        )

    def _parse_evaluation(self, raw_content: str) -> tuple[bool, str]:
        content = raw_content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return False, "Sufficiency evaluator did not return valid JSON."
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return False, "Sufficiency evaluator did not return valid JSON."

        if not isinstance(parsed, dict):
            return False, "Sufficiency evaluator did not return a JSON object."

        return bool(parsed.get("answerable")), str(parsed.get("reason") or "").strip()

    def _conservative_missing_reason(self, query: str, context: str) -> str | None:
        if TEMPORAL_QUERY_PATTERN.search(query):
            if re.search(r"\b(future|upcoming)\b", query, re.IGNORECASE):
                return "Future information not present."
            return "Recent notification or update information not present in retrieved context."

        if MSME_DEFINITION_QUERY_PATTERN.search(query) and "msme" in query.lower():
            if not all(pattern.search(context) for pattern in MSME_THRESHOLD_CONTEXT_PATTERNS):
                return "Definition referenced but actual classification details missing."

        return None

    def _missing_detail_reason(self, query: str, context: str) -> str | None:
        context_lower = context.lower()
        for detail_name, pattern in DETAIL_INTENT_PATTERNS.items():
            if pattern.search(query) and not pattern.search(context_lower):
                return f"{detail_name.title()} not found in retrieved context."
        return None

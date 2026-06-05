import re
from dataclasses import dataclass

from models import GuardrailResult


@dataclass(frozen=True)
class PatternRule:
    name: str
    pattern: re.Pattern[str]
    reason: str


class Guardrails:
    def __init__(self) -> None:
        self.pii_rules = [
            PatternRule(
                "aadhaar",
                re.compile(r"(?<!\d)(?:\d{4}[\s-]?){2}\d{4}(?!\d)"),
                "The input appears to contain an Aadhaar number.",
            ),
            PatternRule(
                "pan",
                re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE),
                "The input appears to contain a PAN number.",
            ),
            PatternRule(
                "email",
                re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
                "The input appears to contain an email address.",
            ),
            PatternRule(
                "phone",
                re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"),
                "The input appears to contain a phone number.",
            ),
            PatternRule(
                "password",
                re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
                "The input appears to contain a password.",
            ),
            PatternRule(
                "api_key",
                re.compile(r"\b(?:api[_-]?key|secret[_-]?key|access[_-]?key)\s*[:=]\s*[A-Za-z0-9_\-]{12,}", re.IGNORECASE),
                "The input appears to contain an API key or secret.",
            ),
            PatternRule(
                "token",
                re.compile(r"\b(?:token|bearer)\s*[:=]?\s*[A-Za-z0-9_\-.]{20,}", re.IGNORECASE),
                "The input appears to contain an access token.",
            ),
            PatternRule(
                "bank_account",
                re.compile(r"\b(?:account|acct|a/c)\s*(?:number|no\.?)?\s*[:=]?\s*\d{9,18}\b", re.IGNORECASE),
                "The input appears to contain a bank account number.",
            ),
        ]
        self.jailbreak_rules = [
            PatternRule(
                "ignore_instructions",
                re.compile(r"\b(ignore|forget|disregard|override)\b.{0,60}\b(previous|prior|above|system|developer|instructions?)\b", re.IGNORECASE),
                "The input attempts to override system or developer instructions.",
            ),
            PatternRule(
                "prompt_extraction",
                re.compile(r"\b(show|reveal|print|display|tell me)\b.{0,60}\b(system prompt|hidden prompt|developer message|configuration|instructions)\b", re.IGNORECASE),
                "The input attempts to extract hidden prompts or configuration.",
            ),
            PatternRule(
                "role_change",
                re.compile(r"\b(act as|pretend to be|you are now|roleplay as|become)\b.{0,80}\b(unrestricted|developer|system|admin|jailbroken|dan)\b", re.IGNORECASE),
                "The input attempts an unsafe role change.",
            ),
            PatternRule(
                "safety_bypass",
                re.compile(r"\b(bypass|disable|remove|turn off)\b.{0,60}\b(safety|guardrails?|filters?|policy|restrictions?)\b", re.IGNORECASE),
                "The input attempts to bypass safety controls.",
            ),
            PatternRule(
                "jailbreak",
                re.compile(r"\b(jailbreak|prompt injection|do anything now|DAN mode)\b", re.IGNORECASE),
                "The input appears to contain a jailbreak or prompt injection attempt.",
            ),
        ]

    def validate(self, query: str) -> GuardrailResult:
        detected: list[str] = []
        reasons: list[str] = []

        for rule in [*self.pii_rules, *self.jailbreak_rules]:
            if rule.pattern.search(query):
                detected.append(rule.name)
                reasons.append(rule.reason)

        if detected:
            return GuardrailResult(
                passed=False,
                detected=detected,
                reason=" ".join(dict.fromkeys(reasons)),
            )
        return GuardrailResult(passed=True)

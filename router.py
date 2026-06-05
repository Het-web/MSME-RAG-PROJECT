import re

from langchain_core.prompts import ChatPromptTemplate

from llm import get_router_llm
from prompts import ROUTER_PROMPT


MSME_KEYWORDS = {
    "msme",
    "startup",
    "entrepreneur",
    "business",
    "udyam",
    "pmegp",
    "cgtmse",
    "mudra",
    "scheme",
    "subsidy",
    "grant",
    "funding",
    "loan",
    "export",
    "import-export",
    "tax",
    "gst",
    "compliance",
    "license",
    "registration",
    "incentive",
    "self-employment",
}

GENERAL_PATTERNS = [
    re.compile(r"\b(weather|movie|recipe|sports|cricket score|relationship|astrology)\b", re.IGNORECASE),
]


class SmartRouter:
    def __init__(self) -> None:
        self.prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)

    def route(self, query: str) -> str:
        lowered = query.lower()
        if any(pattern.search(query) for pattern in GENERAL_PATTERNS):
            return "General"
        if any(keyword in lowered for keyword in MSME_KEYWORDS):
            return "MSME"

        llm = get_router_llm()
        response = (self.prompt | llm).invoke({"query": query})
        route = str(response.content).strip()
        if route.upper().startswith("MSME"):
            return "MSME"
        return "General"

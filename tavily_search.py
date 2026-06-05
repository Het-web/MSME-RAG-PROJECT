import json
import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults

from config import get_settings
from llm import get_router_llm
from models import Source, TavilyResult
from prompts import SEARCH_QUERY_PROMPT
from utils import compact_text, elapsed_ms, now_ms


RECENCY_PATTERNS = [
    re.compile(r"\b(latest|recent|new|newly|current|today|this year|2026|2025)\b", re.IGNORECASE),
    re.compile(r"\b(policy updates?|announcement|budget|launched|amendment|notification|deadline|rate change)\b", re.IGNORECASE),
]


class ConditionalTavilySearch:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.query_prompt = ChatPromptTemplate.from_template(SEARCH_QUERY_PROMPT)
        self._tool: TavilySearchResults | None = None

    def _get_tool(self) -> TavilySearchResults:
        if not self.settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")
        os.environ.setdefault("TAVILY_API_KEY", self.settings.tavily_api_key)
        if self._tool is None:
            self._tool = TavilySearchResults(
                max_results=self.settings.tavily_max_results,
                search_depth=self.settings.tavily_search_depth,
                include_answer=True,
                include_raw_content=False,
                include_images=False,
                exclude_domains=self.settings.tavily_exclude_domains,
            )
        return self._tool

    def should_search(self, query: str) -> tuple[bool, str]:
        if any(pattern.search(query) for pattern in RECENCY_PATTERNS):
            return True, "Query asks for recent, latest, current, or announcement-style information."
        return False, "Query can be answered from the local knowledge base if relevant context exists."

    def search(self, query: str, trigger_reason: str | None = None) -> TavilyResult:
        reason = trigger_reason
        if reason is None:
            should_search, reason = self.should_search(query)
            if not should_search:
                return TavilyResult(used=False, reason=reason)
        if not self.settings.tavily_api_key:
            return TavilyResult(used=False, reason="TAVILY_API_KEY is not configured.")

        metrics: dict[str, float | int] = {}
        rewrite_start = now_ms()
        optimized_query = (self.query_prompt | get_router_llm()).invoke({"query": query}).content.strip()
        metrics["tavily_query_rewrite_ms"] = elapsed_ms(rewrite_start)

        search_start = now_ms()
        result = self._get_tool().invoke({"query": optimized_query})
        metrics["tavily_search_ms"] = elapsed_ms(search_start)

        parsed = self._parse_result(result)
        answer = parsed.get("answer")
        raw_results = parsed.get("results", [])
        context_parts: list[str] = []
        sources: list[Source] = []

        if answer:
            context_parts.append(f"Tavily answer: {answer}")

        for index, item in enumerate(raw_results):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            context_parts.append(f"[Web {index + 1}] {title}\nURL: {url}\n{content}")
            sources.append(
                Source(
                    source_file=url,
                    document_type="web",
                    chunk_id=f"tavily-{index + 1}",
                    content_preview=compact_text(content or title, 240),
                )
            )

        metrics["tavily_results"] = len(raw_results)
        return TavilyResult(
            context="\n\n".join(context_parts),
            sources=sources,
            used=True,
            reason=reason,
            metrics=metrics,
        )

    def _parse_result(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result

        artifact = getattr(result, "artifact", None)
        if isinstance(artifact, dict):
            return artifact

        content = getattr(result, "content", result)
        if isinstance(content, list):
            return {"results": [item for item in content if isinstance(item, dict)]}

        if isinstance(content, str):
            try:
                loaded = json.loads(content)
            except json.JSONDecodeError:
                return {"results": [{"title": "Tavily Search Result", "url": "", "content": content}]}
            if isinstance(loaded, dict):
                return loaded
            if isinstance(loaded, list):
                return {"results": [item for item in loaded if isinstance(item, dict)]}

        return {"results": []}

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config import get_settings


@lru_cache
def get_router_llm() -> BaseChatModel:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for routing.")
    return ChatGroq(
        model=settings.router_model,
        temperature=0,
        api_key=settings.groq_api_key,
        max_tokens=8,
    )


@lru_cache
def get_sufficiency_llm() -> BaseChatModel:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for context sufficiency evaluation.")
    return ChatGroq(
        model=settings.router_model,
        temperature=0,
        api_key=settings.groq_api_key,
        max_tokens=96,
    )


@lru_cache
def get_final_llm() -> BaseChatModel:
    settings = get_settings()
    if settings.final_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required when FINAL_PROVIDER=openrouter.")
        return ChatOpenAI(
            model=settings.final_model,
            temperature=settings.temperature,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=settings.max_output_tokens,
        )

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for final answer generation.")
    return ChatGroq(
        model=settings.final_model or settings.fallback_final_model,
        temperature=settings.temperature,
        api_key=settings.groq_api_key,
        max_tokens=settings.max_output_tokens,
    )

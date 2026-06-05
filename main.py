import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from config import get_settings
from context_sufficiency import ContextSufficiencyEvaluator
from guardrails import Guardrails
from ingestion import PDFIngestionPipeline
from llm import get_final_llm
from models import ChatRequest, ChatResponse, TavilyResult
from prompts import ANSWER_PROMPT, BLOCKED_TEMPLATE, NO_CONTEXT_ANSWER, OUT_OF_SCOPE_TEMPLATE
from retriever import MSMERetriever
from router import SmartRouter
from tavily_search import ConditionalTavilySearch
from utils import configure_logging, elapsed_ms, now_ms, timed

from langchain_core.prompts import ChatPromptTemplate


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

guardrails = Guardrails()
smart_router = SmartRouter()
retriever = MSMERetriever()
sufficiency_evaluator = ContextSufficiencyEvaluator()
tavily = ConditionalTavilySearch()
answer_prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.run_ingestion_on_startup:
        try:
            stats = PDFIngestionPipeline().ingest_all()
            logger.info("ingestion_completed stats=%s", stats)
        except Exception:
            logger.exception("ingestion_failed")
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    total_start = now_ms()
    metrics: dict[str, object] = {}
    query = request.query.strip()

    try:
        with timed(metrics, "guardrail"):
            guard_result = guardrails.validate(query)

        if not guard_result.passed:
            metrics["total_request_ms"] = elapsed_ms(total_start)
            answer = BLOCKED_TEMPLATE.format(reason=guard_result.reason or "Validation failed.")
            logger.info("chat_blocked detected=%s metrics=%s", guard_result.detected, metrics)
            return ChatResponse(route="Blocked", answer=answer, sources=[], metrics=metrics)

        with timed(metrics, "router"):
            route = smart_router.route(query)

        if route == "General":
            metrics["total_request_ms"] = elapsed_ms(total_start)
            answer = OUT_OF_SCOPE_TEMPLATE.format(query=query)
            logger.info("chat_general metrics=%s", metrics)
            return ChatResponse(route="General", answer=answer, sources=[], metrics=metrics)

        retrieval_result = retriever.retrieve(query)
        metrics.update(retrieval_result.metrics)
        local_context = retrieval_result.context
        retrieved_chunks = int(retrieval_result.metrics.get("retrieved_chunks", 0))

        logger.info(
            "retrieval_completed retrieved_chunks=%s context_chars=%s",
            retrieved_chunks,
            len(local_context),
        )

        sufficiency_result = sufficiency_evaluator.evaluate(query, local_context, retrieved_chunks)
        metrics.update(sufficiency_result.metrics)
        logger.info(
            "context_sufficiency_evaluated context_answerable=%s context_sufficiency_reason=%r",
            sufficiency_result.answerable,
            sufficiency_result.reason,
        )

        if sufficiency_result.answerable:
            tavily_result = TavilyResult(used=False, reason="Retrieved context is sufficient.")
            metrics["tavily_trigger_reason"] = ""
        else:
            metrics["tavily_trigger_reason"] = sufficiency_result.reason
            tavily_result = tavily.search(query, trigger_reason=sufficiency_result.reason)

        metrics.update(tavily_result.metrics)
        metrics["tavily_used"] = tavily_result.used
        metrics["tavily_reason"] = tavily_result.reason
        logger.info(
            "tavily_decision tavily_used=%s tavily_trigger_reason=%r",
            tavily_result.used,
            metrics["tavily_trigger_reason"],
        )

        live_context = tavily_result.context
        if not local_context and not live_context:
            metrics["llm_response_generation_ms"] = 0
            metrics["total_request_ms"] = elapsed_ms(total_start)
            return ChatResponse(route="MSME", answer=NO_CONTEXT_ANSWER, sources=[], metrics=metrics)

        with timed(metrics, "llm_response_generation"):
            response = (answer_prompt | get_final_llm()).invoke(
                {
                    "local_context": local_context,
                    "live_context": live_context,
                    "question": query,
                }
            )

        metrics["total_request_ms"] = elapsed_ms(total_start)
        sources = [*retrieval_result.sources, *tavily_result.sources]
        logger.info("chat_success route=%s metrics=%s", route, metrics)
        return ChatResponse(route="MSME", answer=str(response.content), sources=sources, metrics=metrics)

    except RuntimeError as exc:
        metrics["total_request_ms"] = elapsed_ms(total_start)
        logger.exception("chat_runtime_error metrics=%s", metrics)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        metrics["total_request_ms"] = elapsed_ms(total_start)
        logger.exception("chat_unhandled_error metrics=%s", metrics)
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

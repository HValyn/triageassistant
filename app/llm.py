import json

from ollama import ResponseError, chat, list
from pydantic import ValidationError

from app.config import DEFAULT_TRIAGE_MODEL
from app.schemas import LLMTriageOutput, TriageRequest


SYSTEM_PROMPT = """
You are a clinical triage assistant designed for hospital intake.

Your role is NOT to diagnose or recommend treatment.

Your tasks:
1. Classify urgency: emergency | urgent | routine
2. Suggest department
3. Extract red flags
4. Generate a structured summary

Rules:
- Be conservative
- Do not diagnose
- Do not suggest treatment
- Output STRICT JSON only

Schema:
{
  "urgency": "emergency | urgent | routine",
  "department": "string",
  "confidence": 0-1,
  "summary": "string",
  "red_flags": ["string"]
}
""".strip()


def _build_user_prompt(payload: TriageRequest) -> str:
    return (
        f"name={payload.name}\n"
        f"age={payload.age}\n"
        f"gender={payload.gender}\n"
        f"duration={payload.duration}\n"
        f"conditions={payload.conditions}\n"
        f"severity={payload.severity}/10\n"
        f"symptoms={payload.symptoms}"
    )


class ModelNotFoundError(Exception):
    pass


def resolve_model(model: str | None) -> str:
    return (model or DEFAULT_TRIAGE_MODEL).strip()


def list_available_models() -> list[str]:
    try:
        response = list()
        return [m.model for m in response.models]
    except Exception:
        return []


def triage_with_llm(payload: TriageRequest, model: str | None = None) -> LLMTriageOutput:
    resolved_model = resolve_model(model or payload.model)
    try:
        response = chat(
            model=resolved_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(payload)},
            ],
            options={"temperature": 0},
        )
    except ResponseError as exc:
        if exc.status_code == 404:
            raise ModelNotFoundError(
                f"Ollama model '{resolved_model}' not found. Pull it first (ollama pull {resolved_model}) "
                "or choose a different model in the UI."
            ) from exc
        raise
    content = response.message.content.strip()

    try:
        parsed = json.loads(content)
        return LLMTriageOutput.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError):
        fallback = {
            "urgency": "urgent",
            "department": "General Medicine",
            "confidence": 0.5,
            "summary": f"{payload.age}-year-old {payload.gender} with reported symptoms for {payload.duration}.",
            "red_flags": [],
        }
        return LLMTriageOutput.model_validate(fallback)

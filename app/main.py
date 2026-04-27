from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import DEFAULT_TRIAGE_MODEL
from app.conversation import (
    extract_state_from_message,
    generate_follow_up,
    get_missing_fields,
    merge_state,
)
from app.db import init_db, log_triage
from app.llm import ModelNotFoundError, list_available_models, triage_with_llm
from app.rules import apply_rules
from app.schemas import ModelInfoResponse, IntakeTurnRequest, IntakeTurnResponse, TriageRequest, TriageResponse
from app.voice import transcribe_audio_locally


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Clinical Triage Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models", response_model=ModelInfoResponse)
def models() -> ModelInfoResponse:
    return ModelInfoResponse(
        default_model=DEFAULT_TRIAGE_MODEL,
        available_models=list_available_models(),
    )


@app.post("/triage", response_model=TriageResponse)
def triage(payload: TriageRequest) -> TriageResponse:
    try:
        llm_output = triage_with_llm(payload, model=payload.model)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    rule_result = apply_rules(payload, llm_output)

    response = TriageResponse(
        **rule_result.output.model_dump(),
        rule_triggered=",".join(rule_result.triggered_rules) if rule_result.triggered_rules else None,
    )

    log_triage(payload, llm_output, response, rule_result.triggered_rules)
    return response


@app.post("/intake/turn", response_model=IntakeTurnResponse)
def intake_turn(payload: IntakeTurnRequest) -> IntakeTurnResponse:
    try:
        extracted = extract_state_from_message(payload.message, payload.model)
        merged = merge_state(payload.state, extracted)
        missing = get_missing_fields(merged)
        reply = generate_follow_up(merged, missing, payload.model)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IntakeTurnResponse(
        state=merged,
        missing_fields=missing,
        ready_for_triage=len(missing) == 0,
        assistant_reply=reply,
    )


@app.post("/voice/transcribe")
def transcribe_voice(audio: UploadFile = File(...)) -> dict[str, str]:
    transcript = transcribe_audio_locally(audio)
    return {"transcript": transcript}

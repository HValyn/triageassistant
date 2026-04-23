import json

from ollama import chat

from app.schemas import IntakeState


EXTRACTION_PROMPT = """
You are an intake extraction assistant.
Extract and merge patient intake fields from the latest user message.
Return STRICT JSON with keys only:
name, age, gender, duration, conditions, severity, symptoms

Rules:
- Keep unknown fields as null.
- Use short, factual text.
- Never diagnose.
- Never suggest treatment.
""".strip()


FOLLOW_UP_PROMPT = """
You are a clinical triage intake assistant.
Generate one short, empathetic follow-up question to collect missing required intake fields.
Missing fields: {missing_fields}
Current known intake JSON: {state_json}

Rules:
- Ask only one question.
- No diagnosis, no treatment.
- Be concise and polite.
""".strip()


REQUIRED_FIELDS = ["age", "gender", "duration", "severity", "symptoms"]


def _clean_partial_state(parsed: dict) -> IntakeState:
    allowed = {k: parsed.get(k) for k in ["name", "age", "gender", "duration", "conditions", "severity", "symptoms"]}
    return IntakeState.model_validate(allowed)


def merge_state(current: IntakeState, update: IntakeState) -> IntakeState:
    merged = current.model_dump()
    for key, value in update.model_dump().items():
        if value not in (None, ""):
            merged[key] = value
    return IntakeState.model_validate(merged)


def extract_state_from_message(message: str, model: str = "medgemma") -> IntakeState:
    resp = chat(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": message},
        ],
        options={"temperature": 0},
    )
    content = resp.message.content.strip()
    try:
        parsed = json.loads(content)
        return _clean_partial_state(parsed)
    except json.JSONDecodeError:
        return IntakeState()


def get_missing_fields(state: IntakeState) -> list[str]:
    missing = []
    for field in REQUIRED_FIELDS:
        value = getattr(state, field)
        if value in (None, ""):
            missing.append(field)
    return missing


def generate_follow_up(state: IntakeState, missing_fields: list[str], model: str = "medgemma") -> str:
    if not missing_fields:
        return "Thanks — I have enough details to run triage. Please review and submit."
    prompt = FOLLOW_UP_PROMPT.format(
        missing_fields=", ".join(missing_fields),
        state_json=state.model_dump_json(),
    )
    resp = chat(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        options={"temperature": 0.2},
    )
    reply = resp.message.content.strip()
    return reply or "Could you share a little more detail about your symptoms and severity (0-10)?"

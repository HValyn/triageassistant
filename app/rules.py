from dataclasses import dataclass

from app.schemas import LLMTriageOutput, TriageRequest


URGENCY_ORDER = ["routine", "urgent", "emergency"]


@dataclass
class RuleResult:
    output: LLMTriageOutput
    triggered_rules: list[str]


def normalize_text(*parts: str) -> str:
    return " ".join(parts).strip().lower()


def _has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def _has_all(text: str, terms: list[str]) -> bool:
    return all(term in text for term in terms)


def _escalate_once(urgency: str) -> str:
    idx = URGENCY_ORDER.index(urgency)
    return URGENCY_ORDER[min(idx + 1, len(URGENCY_ORDER) - 1)]


def apply_rules(payload: TriageRequest, llm_output: LLMTriageOutput) -> RuleResult:
    text = normalize_text(payload.symptoms, payload.conditions, payload.duration, payload.gender)
    result = llm_output.model_copy(deep=True)
    triggered: list[str] = []

    hard_overrides = [
        {
            "name": "cardiac_emergency",
            "conditions": [["chest pain"], ["sweating", "diaphoresis"], ["radiating", "left arm"]],
            "action": {"urgency": "emergency", "department": "Cardiology"},
        },
        {
            "name": "breathing_distress",
            "conditions": [["difficulty breathing", "shortness of breath", "can't breathe"]],
            "action": {"urgency": "emergency", "department": "Emergency Medicine"},
        },
        {
            "name": "loss_of_consciousness",
            "conditions": [["loss of consciousness", "passed out", "unconscious"]],
            "action": {"urgency": "emergency", "department": "Emergency Medicine"},
        },
        {
            "name": "seizure_red_flag",
            "conditions": [["seizure", "convulsion"]],
            "action": {"urgency": "emergency", "department": "Neurology"},
        },
        {
            "name": "stroke_symptoms",
            "conditions": [["slurred speech", "facial droop", "one-sided weakness", "sudden weakness"]],
            "action": {"urgency": "emergency", "department": "Neurology"},
        },
    ]

    for rule in hard_overrides:
        all_groups_matched = True
        for group in rule["conditions"]:
            if not _has_any(text, group):
                all_groups_matched = False
                break
        if all_groups_matched:
            result.urgency = rule["action"]["urgency"]
            result.department = rule["action"]["department"]
            result.confidence = max(result.confidence, 0.95)
            if rule["name"] not in triggered:
                triggered.append(rule["name"])

    escalation_rules = [
        {
            "name": "child_fever_escalation",
            "when": payload.age <= 12 and _has_any(text, ["fever", "high temperature"]),
        },
        {
            "name": "diabetes_dizziness_escalation",
            "when": _has_any(text, ["diabetes"]) and _has_any(text, ["dizziness", "lightheaded"]),
        },
        {
            "name": "severe_pain_escalation",
            "when": payload.severity >= 8,
        },
    ]

    for rule in escalation_rules:
        if rule["when"]:
            old = result.urgency
            result.urgency = _escalate_once(result.urgency)
            if result.urgency != old:
                triggered.append(rule["name"])

    result.red_flags = sorted(set(result.red_flags))

    if not result.summary:
        result.summary = "Structured triage summary unavailable."

    return RuleResult(output=result, triggered_rules=triggered)

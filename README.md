# AI Clinical Triage Assistant (Local, v1)

Local-first clinical triage assistant built with FastAPI + Streamlit + Ollama (MedGemma), now with conversational intake.

> This system is **not** a diagnostic or treatment tool.

## Features

- Urgency classification: `emergency | urgent | routine`
- Department suggestion
- Red flag extraction
- Structured summary output
- Conversational intake to progressively capture missing fields
- Optional local voice transcription endpoint (when Whisper is installed)
- Configurable LLM model (UI selection + `TRIAGE_MODEL` env default)
- Safety-oriented deterministic rule overrides
- SQLite logging of original and final outcomes

## Tech Stack

- Backend: FastAPI
- LLM runtime: Ollama
- Model: MedGemma (`medgemma`, can be swapped to a specific tag)
- Frontend: Streamlit
- Database: SQLite

## Project Structure

```text
app/
  main.py        # FastAPI app and /triage endpoint
  llm.py         # Ollama + MedGemma prompt integration
  conversation.py# Conversational extraction + follow-up questioning
  rules.py       # Deterministic safety override rules
  voice.py       # Optional local speech-to-text adapter
  db.py          # SQLite table + log writes
  schemas.py     # Pydantic request/response models
streamlit_app.py # Streamlit UI
triage.db        # Created on first backend startup
```

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install and start Ollama (see Ollama docs), then pull MedGemma:

   ```bash
   ollama pull medgemma
   ```

3. Run backend:

   ```bash
   TRIAGE_MODEL=medgemma:4b uvicorn app.main:app --reload --port 8000
   ```

   If omitted, backend defaults to `medgemma:4b`:

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. Run frontend:

   ```bash
   streamlit run streamlit_app.py
   ```

5. (Optional) Enable local voice transcription:

   ```bash
   pip install openai-whisper
   ```

## API

### `POST /triage`

Request example:

```json
{
  "age": 45,
  "gender": "male",
  "duration": "2 hours",
  "conditions": "hypertension",
  "severity": 8,
  "symptoms": "chest pain radiating to left arm, sweating"
}
```

### `POST /intake/turn`

Chat-first intake turn. Sends the user message plus current partially-filled state and returns:
- merged extracted fields
- missing fields list
- one follow-up question
- ready-for-triage boolean

### `GET /models`

Returns backend default model and models currently discoverable from Ollama.

### `POST /voice/transcribe`

Optional local speech-to-text endpoint using Whisper if installed.

Response example:

```json
{
  "urgency": "emergency",
  "department": "Cardiology",
  "confidence": 0.95,
  "summary": "45-year-old male with reported symptoms for 2 hours.",
  "red_flags": [],
  "rule_triggered": "cardiac_emergency,severe_pain_escalation"
}
```

## Safety Model

- No diagnosis language in prompts or outputs
- No treatment recommendations
- Conservative triage bias
- Rule engine can override unsafe LLM outputs
- Every request logged with both LLM and final output
- Missing model errors are returned as clear API `400` errors with guidance to `ollama pull <model>`

## Notes

- Voice support is optional and local-only (requires `openai-whisper` install).
- Image and hospital integrations are still out of scope for this version.

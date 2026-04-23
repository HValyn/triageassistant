import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import LLMTriageOutput, TriageRequest, TriageResponse


DB_PATH = Path("triage.db")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS triage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_json TEXT NOT NULL,
                llm_output TEXT NOT NULL,
                final_output TEXT NOT NULL,
                rules_triggered TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_triage(
    request: TriageRequest,
    llm_output: LLMTriageOutput,
    final_output: TriageResponse,
    rules_triggered: list[str],
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO triage_logs
            (input_json, llm_output, final_output, rules_triggered, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request.model_dump_json(),
                llm_output.model_dump_json(),
                final_output.model_dump_json(),
                json.dumps(rules_triggered),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

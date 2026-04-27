from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import HTTPException, UploadFile


def transcribe_audio_locally(audio: UploadFile) -> str:
    """
    Optional local transcription via whisper package.
    Install with: pip install openai-whisper
    """
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="Voice transcription not enabled. Install `openai-whisper` to use this endpoint.",
        ) from exc

    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio.file.read())
        tmp_path = tmp.name

    model = whisper.load_model("base")
    result = model.transcribe(tmp_path)
    text = (result.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Could not transcribe audio.")
    return text

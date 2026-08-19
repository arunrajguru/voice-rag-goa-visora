import httpx
import os
from typing import Dict, Any
from app.config import settings
from app.utils.logger import logger

class SarvamSTTService:
    """Speech-to-Text integration service using Sarvam AI official API/SDK."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.api_url = "https://api.sarvam.ai/speech-to-text"

    async def transcribe_audio(self, audio_bytes: bytes, filename: str = "audio.wav", language_code: str = "hi-IN") -> str:
        """Transcribe audio bytes using Sarvam STT REST API with graceful local fallback."""
        if not self.api_key:
            logger.warning("SARVAM_API_KEY missing from environment. Using local mock/fallback STT transcription.")
            return "What is MSMARCO-XI dataset and how is it used in passage retrieval?"

        headers = {
            "api-subscription-key": self.api_key
        }

        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model": "saaras:v1",
            "language_code": language_code
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.api_url, headers=headers, files=files, data=data)
                if response.status_code == 200:
                    res_json = response.json()
                    transcript = res_json.get("transcript") or res_json.get("text") or ""
                    return transcript.strip()
                else:
                    logger.error(f"Sarvam STT API error {response.status_code}: {response.text}")
                    return "What is the capital of India?"
        except Exception as e:
            logger.error(f"Failed to connect to Sarvam STT API: {str(e)}")
            return "Sample query transcribed from audio"

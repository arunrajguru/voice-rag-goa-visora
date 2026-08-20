import httpx
from app.config import settings
from app.utils.logger import logger


class SarvamSTTService:
    """Speech-to-Text service using Sarvam AI."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.api_url = "https://api.sarvam.ai/speech-to-text"

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: str = "en-IN"
    ) -> str:

        # Never return a fake/hardcoded question.
        if not self.api_key:
            logger.error("SARVAM_API_KEY is missing.")
            raise RuntimeError("SARVAM_API_KEY is not configured.")

        if not audio_bytes:
            logger.error("Received empty audio.")
            raise ValueError("No audio data received.")

        headers = {
            "api-subscription-key": self.api_key
        }

        # Detect MIME type from filename
        filename_lower = filename.lower()

        if filename_lower.endswith(".webm"):
            mime_type = "audio/webm"
        elif filename_lower.endswith(".mp3"):
            mime_type = "audio/mpeg"
        elif filename_lower.endswith(".m4a"):
            mime_type = "audio/mp4"
        elif filename_lower.endswith(".ogg"):
            mime_type = "audio/ogg"
        elif filename_lower.endswith(".wav"):
            mime_type = "audio/wav"
        else:
            mime_type = "audio/webm"

        files = {
            "file": (
                filename,
                audio_bytes,
                mime_type
            )
        }

        data = {
            "model": "saaras:v1",
            "language_code": language_code
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:

                response = await client.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data
                )

                logger.info(
                    f"Sarvam STT response status: {response.status_code}"
                )

                if response.status_code != 200:
                    logger.error(
                        f"Sarvam STT API error: "
                        f"{response.status_code} - {response.text}"
                    )

                    raise RuntimeError(
                        f"Sarvam STT failed: {response.status_code}"
                    )

                result = response.json()

                logger.info(f"Sarvam STT response: {result}")

                transcript = (
                    result.get("transcript")
                    or result.get("text")
                    or ""
                ).strip()

                if not transcript:
                    logger.error(
                        "Sarvam returned an empty transcript."
                    )
                    raise RuntimeError(
                        "Speech could not be transcribed."
                    )

                logger.info(
                    f"Transcription successful: {transcript}"
                )

                return transcript

        except httpx.TimeoutException:
            logger.error("Sarvam STT request timed out.")
            raise RuntimeError(
                "Speech recognition timed out. Please try again."
            )

        except httpx.RequestError as e:
            logger.error(
                f"Sarvam STT network error: {str(e)}"
            )
            raise RuntimeError(
                "Could not connect to speech recognition service."
            )

        except Exception as e:
            logger.error(
                f"Sarvam STT failed: {str(e)}"
            )
            raise

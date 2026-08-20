import httpx

from app.config import settings
from app.utils.logger import logger


class SarvamSTTService:
    """
    Speech-to-Text service using Sarvam AI REST API.

    The service preserves the actual uploaded audio format
    instead of pretending every recording is WAV.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.SARVAM_API_KEY
        self.api_url = "https://api.sarvam.ai/speech-to-text"

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "recording.webm",
        language_code: str = "unknown"
    ) -> str:

        # --------------------------------------------------
        # Validate API key
        # --------------------------------------------------

        if not self.api_key:
            logger.error(
                "SARVAM_API_KEY is missing from environment."
            )

            raise RuntimeError(
                "SARVAM_API_KEY is not configured on the backend."
            )

        # --------------------------------------------------
        # Validate audio
        # --------------------------------------------------

        if not audio_bytes:
            logger.error(
                "Received empty audio bytes."
            )

            raise ValueError(
                "No audio data received."
            )

        # --------------------------------------------------
        # Determine MIME type
        # --------------------------------------------------

        filename = filename or "recording.webm"

        filename_lower = filename.lower()

        if filename_lower.endswith(".webm"):
            mime_type = "audio/webm"

        elif filename_lower.endswith(".ogg"):
            mime_type = "audio/ogg"

        elif filename_lower.endswith(".opus"):
            mime_type = "audio/opus"

        elif filename_lower.endswith(".mp3"):
            mime_type = "audio/mpeg"

        elif (
            filename_lower.endswith(".m4a")
            or filename_lower.endswith(".mp4")
        ):
            mime_type = "audio/mp4"

        elif filename_lower.endswith(".wav"):
            mime_type = "audio/wav"

        elif filename_lower.endswith(".flac"):
            mime_type = "audio/flac"

        elif filename_lower.endswith(".aac"):
            mime_type = "audio/aac"

        else:
            # Browser recordings from MediaRecorder are
            # normally WebM/Opus.
            mime_type = "audio/webm"

        logger.info(
            f"Preparing Sarvam STT request: "
            f"filename={filename}, "
            f"mime_type={mime_type}, "
            f"size={len(audio_bytes)} bytes, "
            f"language={language_code}"
        )

        # --------------------------------------------------
        # Headers
        # --------------------------------------------------

        headers = {
            "api-subscription-key": self.api_key
        }

        # --------------------------------------------------
        # Multipart audio file
        # --------------------------------------------------

        files = {
            "file": (
                filename,
                audio_bytes,
                mime_type
            )
        }

        # --------------------------------------------------
        # Sarvam STT parameters
        # --------------------------------------------------

        data = {
            "model": "saaras:v3",
            "language_code": language_code,
            "mode": "transcribe"
        }

        # --------------------------------------------------
        # Call Sarvam
        # --------------------------------------------------

        try:

            async with httpx.AsyncClient(
                timeout=30.0
            ) as client:

                response = await client.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data
                )

            logger.info(
                f"Sarvam STT HTTP status: "
                f"{response.status_code}"
            )

            # --------------------------------------------------
            # Error response
            # --------------------------------------------------

            if response.status_code != 200:

                logger.error(
                    "Sarvam STT API error: "
                    f"status={response.status_code}, "
                    f"response={response.text}"
                )

                # IMPORTANT:
                # Never return a fake question here.
                raise RuntimeError(
                    "Sarvam Speech-to-Text failed "
                    f"with status {response.status_code}: "
                    f"{response.text}"
                )

            # --------------------------------------------------
            # Parse response
            # --------------------------------------------------

            try:
                result = response.json()

            except Exception as json_error:

                logger.error(
                    f"Could not parse Sarvam response: "
                    f"{json_error}"
                )

                raise RuntimeError(
                    "Invalid response received from Sarvam STT."
                )

            logger.info(
                f"Sarvam STT response: {result}"
            )

            # --------------------------------------------------
            # Extract transcript
            # --------------------------------------------------

            transcript = (
                result.get("transcript")
                or result.get("text")
                or ""
            )

            transcript = transcript.strip()

            # --------------------------------------------------
            # Empty transcript
            # --------------------------------------------------

            if not transcript:

                logger.error(
                    "Sarvam returned an empty transcript."
                )

                raise RuntimeError(
                    "Sarvam could not detect speech in the audio."
                )

            # --------------------------------------------------
            # Success
            # --------------------------------------------------

            logger.info(
                f"Sarvam transcription successful: "
                f"{transcript}"
            )

            return transcript

        # ------------------------------------------------------
        # Timeout
        # ------------------------------------------------------

        except httpx.TimeoutException as error:

            logger.error(
                f"Sarvam STT request timed out: {error}"
            )

            raise RuntimeError(
                "Speech recognition timed out. "
                "Please try again."
            )

        # ------------------------------------------------------
        # Network error
        # ------------------------------------------------------

        except httpx.RequestError as error:

            logger.error(
                f"Sarvam STT network error: {error}"
            )

            raise RuntimeError(
                "Could not connect to Sarvam "
                "Speech-to-Text service."
            )

        # ------------------------------------------------------
        # Re-raise our own errors
        # ------------------------------------------------------

        except RuntimeError:
            raise

        except ValueError:
            raise

        # ------------------------------------------------------
        # Unexpected error
        # ------------------------------------------------------

        except Exception as error:

            logger.exception(
                f"Unexpected Sarvam STT error: {error}"
            )

            raise RuntimeError(
                f"Speech recognition failed: {error}"
            )

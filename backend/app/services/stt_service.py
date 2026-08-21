import httpx

from app.config import settings
from app.utils.logger import logger


class SarvamSTTService:
    """
    Speech-to-Text service using Sarvam AI REST API.

    Supports browser-recorded audio such as:
    - WebM / Opus
    - OGG
    - MP4 / M4A
    - WAV
    - MP3
    - FLAC
    - AAC
    """

    def __init__(self, api_key: str = None):

        self.api_key = (
            api_key
            or settings.SARVAM_API_KEY
        )

        self.api_url = (
            "https://api.sarvam.ai/speech-to-text"
        )

    # ======================================================
    # MIME TYPE DETECTION
    # ======================================================

    def _get_mime_type(
        self,
        filename: str
    ) -> str:

        filename = (
            filename
            or "recording.webm"
        ).lower()

        if filename.endswith(".webm"):
            return "audio/webm"

        if filename.endswith(".ogg"):
            return "audio/ogg"

        if filename.endswith(".opus"):
            return "audio/opus"

        if filename.endswith(".wav"):
            return "audio/wav"

        if filename.endswith(".mp3"):
            return "audio/mpeg"

        if filename.endswith(".m4a"):
            return "audio/mp4"

        if filename.endswith(".mp4"):
            return "audio/mp4"

        if filename.endswith(".flac"):
            return "audio/flac"

        if filename.endswith(".aac"):
            return "audio/aac"

        # Browser MediaRecorder normally produces
        # WebM/Opus on Chrome/Android.
        return "audio/webm"

    # ======================================================
    # TRANSCRIBE AUDIO
    # ======================================================

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "recording.webm",
        language_code: str = "unknown"
    ) -> str:

        # ==================================================
        # 1. CHECK API KEY
        # ==================================================

        if not self.api_key:

            logger.error(
                "SARVAM_API_KEY is missing."
            )

            raise RuntimeError(
                "SARVAM_API_KEY is not configured on the backend."
            )

        # ==================================================
        # 2. CHECK AUDIO
        # ==================================================

        if not audio_bytes:

            logger.error(
                "Received empty audio."
            )

            raise ValueError(
                "No audio data received."
            )

        # ==================================================
        # 3. FILE INFORMATION
        # ==================================================

        filename = (
            filename
            or "recording.webm"
        )

        mime_type = self._get_mime_type(
            filename
        )

        logger.info(
            "========== SARVAM STT REQUEST =========="
        )

        logger.info(
            f"Filename: {filename}"
        )

        logger.info(
            f"MIME type: {mime_type}"
        )

        logger.info(
            f"Audio size: {len(audio_bytes)} bytes"
        )

        logger.info(
            f"Language code: {language_code}"
        )

        # ==================================================
        # 4. HEADERS
        # ==================================================

        headers = {
            "api-subscription-key": self.api_key,
            "Accept": "application/json"
        }

        # ==================================================
        # 5. MULTIPART FILE
        # ==================================================

        files = {
            "file": (
                filename,
                audio_bytes,
                mime_type
            )
        }

        # ==================================================
        # 6. SARVAM PARAMETERS
        # ==================================================

        data = {
            "model": "saaras:v3",
            "language_code": language_code,
            "mode": "transcribe"
        }

        logger.info(
            "Sending audio to Sarvam..."
        )

        # ==================================================
        # 7. HTTP TIMEOUT
        # ==================================================

        timeout = httpx.Timeout(
            connect=15.0,
            read=45.0,
            write=45.0,
            pool=15.0
        )

        # ==================================================
        # 8. CALL SARVAM
        # ==================================================

        try:

            async with httpx.AsyncClient(
                timeout=timeout
            ) as client:

                response = await client.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data
                )

            # ==================================================
            # 9. LOG RESPONSE
            # ==================================================

            logger.info(
                f"Sarvam HTTP status: "
                f"{response.status_code}"
            )

            logger.info(
                "Sarvam response body: "
                f"{response.text[:2000]}"
            )

            # ==================================================
            # 10. HANDLE API ERROR
            # ==================================================

            if response.status_code != 200:

                logger.error(
                    "Sarvam STT API failed."
                )

                raise RuntimeError(
                    "Sarvam Speech-to-Text failed "
                    f"with status {response.status_code}: "
                    f"{response.text[:1000]}"
                )

            # ==================================================
            # 11. PARSE JSON
            # ==================================================

            try:

                result = response.json()

            except Exception as error:

                logger.error(
                    f"Could not parse Sarvam response: {error}"
                )

                raise RuntimeError(
                    "Invalid response received from Sarvam STT."
                )

            logger.info(
                f"Sarvam JSON response: {result}"
            )

            # ==================================================
            # 12. GET TRANSCRIPT
            # ==================================================

            transcript = (
                result.get("transcript")
                or result.get("text")
                or ""
            )

            transcript = transcript.strip()

            # ==================================================
            # 13. EMPTY TRANSCRIPT
            # ==================================================

            if not transcript:

                logger.error(
                    "Sarvam returned an empty transcript."
                )

                raise RuntimeError(
                    "Sarvam could not detect speech in the audio."
                )

            # ==================================================
            # 14. SUCCESS
            # ==================================================

            logger.info(
                "========== SARVAM STT SUCCESS =========="
            )

            logger.info(
                f"Transcript: {transcript}"
            )

            return transcript

        # ==================================================
        # 15. TIMEOUT
        # ==================================================

        except httpx.TimeoutException as error:

            logger.error(
                f"Sarvam STT timeout: {error}"
            )

            raise RuntimeError(
                "Speech recognition timed out. "
                "Please record a shorter voice query "
                "and try again."
            )

        # ==================================================
        # 16. NETWORK ERROR
        # ==================================================

        except httpx.RequestError as error:

            logger.error(
                f"Sarvam STT network error: {error}"
            )

            raise RuntimeError(
                "Could not connect to Sarvam "
                "Speech-to-Text service."
            )

        # ==================================================
        # 17. OUR OWN ERRORS
        # ==================================================

        except RuntimeError:

            raise

        except ValueError:

            raise

        # ==================================================
        # 18. UNEXPECTED ERROR
        # ==================================================

        except Exception as error:

            logger.exception(
                f"Unexpected Sarvam STT error: {error}"
            )

            raise RuntimeError(
                f"Speech recognition failed: {error}"
            )

"""
tts_stt.py - Text-to-Speech utilities (STT handled in app.py via JS)
"""

import io
import base64
from typing import Optional


def text_to_speech_base64(text: str, lang: str = "en") -> Optional[str]:
    """Convert text to speech and return base64 encoded audio."""
    try:
        from gtts import gTTS
        # Truncate long answers for TTS
        tts_text = text[:600]
        tts = gTTS(text=tts_text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except ImportError:
        return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None


def get_audio_html(audio_b64: str) -> str:
    return f"""
    <audio autoplay controls style="width:100%;border-radius:10px;margin-top:8px;accent-color:#2d8a4e">
        <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
    </audio>
    """
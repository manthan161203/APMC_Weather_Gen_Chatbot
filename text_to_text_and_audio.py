# text_to_text_and_audio.py

from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid, os

from utils import (
    detect_text_language,            # text → language
    translate_text,                  # text → translated text
    convert_text_to_speech           # text → speech
)

# Import the new invoke function
from agent import invoke_agent

router = APIRouter()

# ---------------------------------------------------------------------
# GET /get-audio/{filename} → Serve generated MP3 file
# ---------------------------------------------------------------------
@router.get("/get-audio/{filename}", tags=["Text to Text and Audio"])
def get_audio(filename: str):
    audio_path = os.path.join("outputs", filename)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ---------------------------------------------------------------------
# POST /text → run agent → (translate) → TTS → return response + audio
# ---------------------------------------------------------------------
class TextQuery(BaseModel):
    text: str
    langs: List[str] = []

@router.post("/text", tags=["Text to Text and Audio"])
async def text_to_text_and_audio(
    request: Request,
    payload: TextQuery,
    lat: Optional[float] = Query(default=None, description="Latitude (optional)"),
    lon: Optional[float] = Query(default=None, description="Longitude (optional)"),
):
    print("Received text query request")

    user_text = payload.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty text provided")

    # ---------------- Run the agent with session memory ---------------
    try:
        session_id = request.query_params.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())

        print(f"Using session ID: {session_id}")
        print(f"User input: {user_text}")
        
        # Use the new invoke function with proper memory
        result = invoke_agent(session_id, user_text, lat, lon)
        answer_text = result["output"]

        print("Agent result type:", type(answer_text))
        print("Agent result:", answer_text)

    except Exception as e:
        print("Agent error:", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # ---------------- Detect language and optionally translate ------------
    try:
        user_lang = detect_text_language(user_text)
        print("Detected user language:", user_lang)

        # # Match allowed language (if any)
        # if payload.langs:
        #     for lang in payload.langs:
        #         if lang.startswith(user_lang.split("-")[0]):
        #             user_lang = lang
        #             break

        # Match allowed language (if any)
        if payload.langs:
            matched = False
            for lang in payload.langs:
                if lang.startswith(user_lang.split("-")[0]):
                    user_lang = lang
                    matched = True
                    break
            if not matched:
                # Fallback to the first allowed language
                user_lang = payload.langs[0]
                print(f"No match found, using fallback language: {user_lang}")

         # if user_lang != "en-IN":
        #     answer_text = translate_text(
        #         text=answer_text,
        #         target_language_code=user_lang,
        #         source_language_code="en-IN"
        #     )
        
        # Translate if output language is different from expected
        detected_answer_lang = detect_text_language(answer_text)
        print(f"Detected answer language: {detected_answer_lang}")
        
        if detected_answer_lang != user_lang:
            try:
                answer_text = translate_text(
                    text=answer_text,
                    target_language_code=user_lang,
                    source_language_code=detected_answer_lang
                )
                print("Translated answer text:", answer_text)
            except Exception as e:
                print("Translation failed:", e)

    except Exception as e:
        print("Language detection/translation failed:", e)
        user_lang = "en-IN"

    # ---------------- Text → speech ---------------------------------------
    os.makedirs("outputs", exist_ok=True)
    audio_filename = f"{uuid.uuid4()}.mp3"
    output_path = os.path.join("outputs", audio_filename)
    try:
        convert_text_to_speech(
            text=answer_text,
            language_code=user_lang if user_lang else "en-IN",
            output_file_path=output_path
        )
        print(f"Generated TTS audio at {output_path}")
    except Exception as e:
        print("TTS error:", e)
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

    # ---------------- Return JSON response --------------------------------
    audio_url = str(request.base_url) + f"get-audio/{audio_filename}"
    print("Returning response with audio_url:", audio_url)
    return {
        "text": answer_text,
        "audio_url": audio_url,
        "language": user_lang,
        "audio_filename": audio_filename,
        "session_id": session_id
    }
# audio_to_text_and_audio.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
import os, uuid, shutil

from utils import (
    convert_speech_to_text,          # speech → text
    convert_text_to_speech,          # text → speech
    detect_text_language,            # detect language
    translate_text,                  # translate response
    validate_audio_duration          # ensure ≤ 20 sec
)

# Import the new invoke function
from agent import invoke_agent

router = APIRouter()

# ---------------------------------------------------------------------
# GET /get-audio/{filename} → Serve generated MP3 file
# ---------------------------------------------------------------------
@router.get("/get-audio/{filename}", tags=["Audio to Text and Audio"])
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
# POST /audio → transcribe → run agent → TTS → return response + audio
# ---------------------------------------------------------------------
@router.post("/audio", tags=["Audio to Text and Audio"])
async def upload_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    lat: float | None = None,
    lon: float | None = None,
):
    print("Received audio upload request")

    # ---------------- Validate and save uploaded file -----------------------
    if not (audio_file.filename.endswith(".mp3") or audio_file.filename.endswith(".wav")):
        print("Invalid file type:", audio_file.filename)
        raise HTTPException(status_code=400, detail="Only .mp3 or .wav files are allowed")

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{uuid.uuid4()}_{audio_file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        print(f"Saved uploaded file to {file_path}")
    finally:
        audio_file.file.close()

    # ---------------- Check duration limit ----------------------------
    if not validate_audio_duration(file_path):
        print("Audio duration exceeds 20 seconds")
        raise HTTPException(
            status_code=400,
            detail="Audio duration exceeds 20 seconds. Please upload a shorter clip."
        )

    # ---------------- Speech → text -----------------------------------
    try:
        user_text = convert_speech_to_text(file_path)
        print("Transcribed text:", user_text)
        
    except Exception as e:
        print("Speech-to-text error:", e)
        raise HTTPException(status_code=500, detail=f"Speech-to-text error: {e}")

    # ---------------- Run the agent with session memory ---------------
    try:
        session_id = request.query_params.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())

        print(f"Using session ID: {session_id}")
        
        # Use the new invoke function with proper memory
        result = invoke_agent(session_id, user_text, lat, lon)
        answer_text = result["output"]
        
        print("Agent result type:", type(answer_text))
        print("Agent result:", answer_text)

    except Exception as e:
        print("Agent error:", e)
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # ---------------- Detect language and optionally translate --------
    try:
        user_lang = detect_text_language(user_text)
        print("Detected user language:", user_lang)

        if user_lang != "en-IN":
            answer_text = translate_text(
                text=answer_text,
                target_language_code=user_lang,
                source_language_code="en-IN"
            )
            print("Translated answer text:", answer_text)

    except Exception as e:
        print("Language detection/translation failed:", e)
        user_lang = "en-IN"

    # ---------------- Text → speech -----------------------------------
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

    # ---------------- Cleanup uploaded file ---------------------------
    try:
        os.remove(file_path)
        print(f"Cleaned up uploaded file: {file_path}")
    except Exception as e:
        print(f"Could not remove uploaded file: {e}")

    # ---------------- Return response ---------------------------------
    audio_url = str(request.base_url) + f"get-audio/{audio_filename}"
    print("Returning response with audio_url:", audio_url)
    return {
        "text": answer_text,
        "audio_url": audio_url,
        "language": user_lang,
        "audio_filename": audio_filename,
        "session_id": session_id
    }
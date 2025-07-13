from fastapi import FastAPI
from audio_to_text_and_audio import router as audio_router

app = FastAPI(
    title="APMC Voice Chatbot",
    description="Handles voice queries for weather, mandi prices, diseases, and crop suggestions.",
    version="1.0.0"
)

# Include your router
app.include_router(audio_router,tags=["Audio to Text and Audio"])

# Optional: health check endpoint
@app.get("/")
def read_root():
    return {"status": "ok", "message": "APMC Chatbot API is running."}

from fastapi import FastAPI
from app.config import get_settings
from app.routers import chat, audio, quiz


app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().app_version,
    description="AI Assistant microservice for IBLOG e-learning platform",
)

app.include_router(chat.router)
app.include_router(audio.router)
app.include_router(quiz.router)


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "version": get_settings().app_version}

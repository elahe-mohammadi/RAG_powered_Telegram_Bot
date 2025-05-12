# from pathlib import Path
# from dotenv import load_dotenv

# # Load the .env in project root:
# env_path = Path(__file__).parent.parent / ".env"
# load_dotenv(env_path)

import threading
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings, SettingsConfigDict

from .api.rag import router as rag_router
from .services.telegram_bot import TelegramSettings, run_telegram_bot

# ----------------------------------------
# Logging
# ----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------
# App-wide Settings
# ----------------------------------------
class AppSettings(BaseSettings):
    app_name: str = "RAG-Powered Support Bot"
    debug: bool = False
    enable_telegram: bool = True

    model_config = SettingsConfigDict(
        env_prefix = "RAG_APP",
        env_file = ".env",
        extra="ignore",)

settings = AppSettings()

# ----------------------------------------
# Lifespan: startup & shutdown logic
# ----------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.enable_telegram:
        try:
            logger.info("Starting Telegram bot…")
            tg_settings = TelegramSettings()
            thread = threading.Thread(
                target=run_telegram_bot,
                args=(tg_settings,),
                daemon=True,
            )
            thread.start()
            logger.info("✅ Telegram bot started")
        except Exception as e:
            logger.exception("❌ Failed to start Telegram bot")

    yield

    # Shutdown (optional)
    # You could cleanly stop the Telegram thread or close resources here
    logger.info("Shutting down…")

# ----------------------------------------
# Create FastAPI app
# ----------------------------------------
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# CORS (lock down in prod!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Mount your RAG router
app.include_router(
    rag_router,
    prefix="/api",
    tags=["rag"],
)

# Health check
@app.get("/", tags=["health"])
async def health():
    return {"status": "ok", "service": settings.app_name}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

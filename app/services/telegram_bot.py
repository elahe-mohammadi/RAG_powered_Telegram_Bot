

import logging
from pydantic_settings import BaseSettings, SettingsConfigDict
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import httpx

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

class TelegramSettings(BaseSettings):
    bot_token: str
    api_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_prefix = "RAG_TELEGRAM_",
        env_file = ".env",
        extra="ignore",
    )

class TelegramBot:
    def __init__(self, settings: TelegramSettings):
        self.settings = settings
        self.http = httpx.AsyncClient(timeout=10.0)

        self.app = (
            Application.builder()
            .token(settings.bot_token)
            .build()
        )
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help",  self.help_command))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        self.app.add_error_handler(self.error_handler)

    async def start_command(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "👋 Welcome to the RAG Support Bot!\n"
            "Ask me anything and I'll search my documents to help you."
        )

    async def help_command(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "💡 How to use:\n"
            "• Send any question and get an answer.\n"
            "• Append the word “sources” to see where I found it.\n\n"
            "Commands:\n"
            "/start - Restart the bot\n"
            "/help - Show this message"
        )

    async def handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        await ctx.bot.send_chat_action(chat_id, action="typing")

        question = update.message.text.strip()
        show_sources = "sources" in question.lower()

        try:
            # print(f"{self.settings.api_url}")
            resp = await self.http.post(
                f"{self.settings.api_url}/api/ask",
                json={"question": question},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("API call failed: %s", e)
            return await update.message.reply_text(
                "⚠️ Sorry, I'm having trouble right now. Try again later."
            )

        answer = data.get("answer", "I couldn't generate an answer.")
        if show_sources and data.get("sources"):
            valid = [s for s in data["sources"] if s]
            if valid:
                answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in valid)

        await update.message.reply_text(answer)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.error("Telegram update error: %s", context.error)
        if update.effective_message:
            await update.effective_message.reply_text(
                "❌ Sorry, something went wrong. Please try again."
            )

    def run_polling(self):
        self.app.run_polling()

    async def run_webhook(self, webhook_url: str, port: int = 8443):
        await self.app.bot.set_webhook(webhook_url)
        await self.app.start()

import asyncio
def run_telegram_bot(settings: TelegramSettings):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    bot = TelegramBot(settings)
    bot.run_polling()

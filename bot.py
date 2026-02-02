# DeepSeek — InlineAdmin v0.80
#
# AI чат-бот с поддержкой нескольких режимов общения и генерации изображений
# Requirements: python-telegram-bot==20.6, aiohttp==3.9.5, python-dotenv==1.0.1

import os, json, logging, aiohttp, asyncio, time, base64, secrets, string
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List, Set, Tuple
from telegram import (
    Update, KeyboardButton, ReplyKeyboardMarkup,
    InlineKeyboardButton, InlineKeyboardMarkup, InputFile
)
from telegram.error import BadRequest, Forbidden, NetworkError, TimedOut
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

logger = logging.getLogger(__name__)

VERSION = "v0.90"

# ========== RATE LIMITING ==========
RATE_LIMIT_MESSAGES = 30  # Максимум сообщений
RATE_LIMIT_WINDOW = 60    # За сколько секунд (60 = 1 минута)
RATE_LIMIT_IMAGES = 10    # Максимум картинок в минуту
RATE_LIMIT_COOLDOWN = 30  # Время блокировки в секундах

# ========== RETRY SETTINGS ==========
RETRY_MAX_ATTEMPTS = 3    # Максимум попыток при ошибке
RETRY_BASE_DELAY = 1.0    # Начальная задержка (секунды)
RETRY_MAX_DELAY = 10.0    # Максимальная задержка

# ========== ADMIN NOTIFICATIONS ==========
NOTIFY_NEW_USERS = True           # Уведомлять о новых пользователях
NOTIFY_ERRORS = True              # Уведомлять о критических ошибках
NOTIFY_NEW_USERS_THRESHOLD = 10   # Уведомлять каждые N новых пользователей

# ========== USER STATS STORAGE ==========
USER_STATS_FILE = "user_stats.json"

# Хранилище rate-limit данных
rate_limit_data: Dict[int, Dict[str, Any]] = {}

# Хранилище оценок ответов
feedback_data: Dict[str, Dict[str, Any]] = {}

# Модель для хранения сообщений (в памяти)
class ChatMessage:
    def __init__(self, role: str, user_id: int, chat_id: int, text: str, timestamp: datetime = None):
        self.role = role  # 'user' или 'bot'
        self.user_id = user_id
        self.chat_id = chat_id
        self.text = text
        self.timestamp = timestamp or datetime.now()

    def to_dict(self):
        return {
            "role": self.role,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "text": self.text,
            "timestamp": self.timestamp.isoformat()
        }

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DEFAULT_MODEL  = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat-v3-0324")
ADMIN_ID       = int(os.getenv("ADMIN_ID", "0"))
IMAGE_MODEL    = os.getenv("IMAGE_MODEL", "google/gemini-2.5-flash-image-preview")

CODES_FILE = "codes.json"
CUSTOM_PROMPTS_FILE = "custom_prompts.json"


def _parse_api_keys(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    keys: List[str] = []
    normalized = raw.replace("\n", ",").replace(";", ",")
    for chunk in normalized.split(","):
        cleaned = chunk.strip().strip('"').strip("'")
        if cleaned:
            keys.append(cleaned)
    return keys


AI_API_KEYS = _parse_api_keys(os.getenv("AI_API_KEYS"))
if not AI_API_KEYS:
    fallback = os.getenv("AI_API_KEY")
    if fallback:
        AI_API_KEYS = _parse_api_keys(fallback)


class APIKeyManager:
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("APIKeyManager requires at least one API key.")
        self._entries: List[Dict[str, Any]] = [
            {
                "value": key,
                "errors": 0,
                "success": 0,
                "total": 0,
                "last_status": None,
                "last_error": None,
                "last_updated": None,
                "last_used": 0,
            }
            for key in keys
        ]
        self._lock = asyncio.Lock()
        self._usage_tick = 0

    @property
    def count(self) -> int:
        return len(self._entries)

    def has_alternatives(self) -> bool:
        return self.count > 1

    def _ordered_indices(self, exclude: Optional[Set[int]] = None) -> List[int]:
        exclude = exclude or set()
        indices = [i for i in range(len(self._entries)) if i not in exclude]
        if not indices:
            indices = list(range(len(self._entries)))
        indices.sort(key=lambda i: (self._entries[i]["errors"], self._entries[i]["total"], self._entries[i]["last_used"]))
        return indices

    async def acquire(self, exclude: Optional[Set[int]] = None) -> Tuple[str, int]:
        async with self._lock:
            order = self._ordered_indices(exclude)
            idx = order[0]
            entry = self._entries[idx]
            self._usage_tick += 1
            entry["last_used"] = self._usage_tick
            entry["total"] += 1
            return entry["value"], idx

    async def record_success(self, idx: int) -> None:
        async with self._lock:
            entry = self._entries[idx]
            entry["success"] += 1
            entry["last_status"] = 200
            entry["last_error"] = None
            entry["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def record_error(self, idx: int, status: Any, message: Optional[str] = None) -> None:
        async with self._lock:
            entry = self._entries[idx]
            entry["errors"] += 1
            entry["last_status"] = status
            entry["last_error"] = (message or "").strip()[:200] or None
            entry["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    async def snapshot(self) -> List[Dict[str, Any]]:
        async with self._lock:
            order = self._ordered_indices()
            best_idx = order[0] if order else None
            result: List[Dict[str, Any]] = []
            for idx in order:
                entry = self._entries[idx]
                total = entry["total"]
                errors = entry["errors"]
                error_rate = (errors / total) if total else 0.0
                result.append(
                    {
                        "index": idx,
                        "display": self._mask_key(entry["value"]),
                        "total": total,
                        "errors": errors,
                        "success": entry["success"],
                        "last_status": entry["last_status"],
                        "last_error": entry["last_error"],
                        "last_updated": entry["last_updated"],
                        "error_rate": error_rate,
                        "preferred": idx == best_idx,
                    }
                )
            return result

    @staticmethod
    def _mask_key(value: str) -> str:
        if len(value) <= 12:
            return value
        return f"{value[:8]}…{value[-4:]}"


if not AI_API_KEYS:
    raise RuntimeError("AI API keys are not configured. Add AI_API_KEYS or AI_API_KEY to .env.")

AI_KEY_MANAGER = APIKeyManager(AI_API_KEYS)

CODES_LOCK = asyncio.Lock()
CUSTOM_PROMPTS_LOCK = asyncio.Lock()


def _load_json_file(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
    return default


def _save_json_file(path: str, data) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save {path}: {e}")


def load_codes_data() -> Dict[str, Dict[str, Any]]:
    data = _load_json_file(CODES_FILE, {})
    if isinstance(data, list):
        # migrate legacy list format if ever existed
        converted = {}
        for item in data:
            if isinstance(item, dict) and "code" in item:
                converted[item["code"]] = item
        data = converted
    if not isinstance(data, dict):
        data = {}
    return data


def load_custom_profiles() -> Dict[str, Dict[str, Any]]:
    data = _load_json_file(CUSTOM_PROMPTS_FILE, {})
    if not isinstance(data, dict):
        data = {}
    return data


REDEMPTION_CODES: Dict[str, Dict[str, Any]] = load_codes_data()
CUSTOM_PROFILES: Dict[str, Dict[str, Any]] = load_custom_profiles()


def get_custom_profile(user_id: int) -> Optional[Dict[str, Any]]:
    return CUSTOM_PROFILES.get(str(user_id))


def user_has_custom_access(user_id: int) -> bool:
    profile = get_custom_profile(user_id)
    return bool(profile and profile.get("code"))


def user_has_custom_prompt(user_id: int) -> bool:
    profile = get_custom_profile(user_id)
    return bool(profile and profile.get("prompt"))


def _generate_unique_code(length: int = 12) -> str:
    alphabet = string.ascii_uppercase + string.digits
    while True:
        code = "".join(secrets.choice(alphabet) for _ in range(length))
        if code not in REDEMPTION_CODES:
            return code


def _model_value_from_key(key: str) -> Optional[str]:
    return MODEL_MAP.get(key)


def _model_key_from_value(value: str) -> Optional[str]:
    for key, val in MODEL_MAP.items():
        if val == value:
            return key
    return None


async def create_redeem_code(admin_id: int, comment: Optional[str] = None) -> Dict[str, Any]:
    async with CODES_LOCK:
        code = _generate_unique_code()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "code": code,
            "status": "active",
            "created_at": now,
            "created_by": admin_id,
            "comment": comment or "",
            "used_by": None,
            "used_at": None,
            "revoked_by": None,
            "revoked_at": None,
        }
        REDEMPTION_CODES[code] = entry
        _save_json_file(CODES_FILE, REDEMPTION_CODES)
        return entry


async def revoke_code(code: str, admin_id: int) -> Tuple[bool, str]:
    async with CODES_LOCK:
        entry = REDEMPTION_CODES.get(code)
        if not entry:
            return False, "Код не найден."
        if entry.get("status") == "used":
            return False, "Нельзя отозвать уже использованный код."
        if entry.get("status") == "revoked":
            return False, "Код уже отозван."
        entry["status"] = "revoked"
        entry["revoked_by"] = admin_id
        entry["revoked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_json_file(CODES_FILE, REDEMPTION_CODES)
        return True, "Код отозван."


async def redeem_code_for_user(user_id: int, code: str) -> Tuple[bool, str]:
    code = code.strip().upper()
    async with CODES_LOCK:
        if user_has_custom_access(user_id):
            return False, "У тебя уже активирован персональный режим."
        entry = REDEMPTION_CODES.get(code)
        if not entry:
            return False, "Код не найден. Проверь написание."
        if entry.get("status") != "active":
            status = entry.get("status")
            if status == "used":
                return False, "Этот код уже использован."
            if status == "revoked":
                return False, "Этот код отменён админом."
            return False, "Код недоступен."
        entry["status"] = "used"
        entry["used_by"] = user_id
        entry["used_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_json_file(CODES_FILE, REDEMPTION_CODES)

    async with CUSTOM_PROMPTS_LOCK:
        profile = CUSTOM_PROFILES.setdefault(str(user_id), {})
        profile.setdefault("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        profile["code"] = code
        profile["redeemed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profile.setdefault("model", MODEL_MAP["ds"])
        profile.setdefault("title", None)
        profile.setdefault("prompt", None)
        profile.setdefault("updated_at", None)
        _save_json_file(CUSTOM_PROMPTS_FILE, CUSTOM_PROFILES)
    return True, "Код активирован! Теперь можешь создать личный режим."


async def update_custom_prompt(user_id: int, title: str, prompt: str) -> Tuple[bool, str]:
    if not user_has_custom_access(user_id):
        return False, "Сначала активируй код."
    title = title.strip()
    prompt = prompt.strip()
    if not (1 <= len(title) <= 48):
        return False, "Название должно быть от 1 до 48 символов."
    if len(prompt) < 10:
        return False, "Промпт слишком короткий."

    async with CUSTOM_PROMPTS_LOCK:
        profile = CUSTOM_PROFILES.setdefault(str(user_id), {})
        profile["title"] = title
        profile["prompt"] = prompt
        profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profile.setdefault("model", MODEL_MAP["ds"])
        profile.setdefault("created_at", profile.get("updated_at"))
        _save_json_file(CUSTOM_PROMPTS_FILE, CUSTOM_PROFILES)
    return True, "Персональный режим сохранён."


async def delete_custom_prompt(user_id: int) -> Tuple[bool, str]:
    profile = get_custom_profile(user_id)
    if not profile or not profile.get("prompt"):
        return False, "Персональный режим уже пуст."
    async with CUSTOM_PROMPTS_LOCK:
        profile = CUSTOM_PROFILES.setdefault(str(user_id), {})
        profile["prompt"] = None
        profile["title"] = None
        profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_json_file(CUSTOM_PROMPTS_FILE, CUSTOM_PROFILES)
    return True, "Персональный режим очищен. Можешь создать новый."


async def set_custom_model(user_id: int, model_key: str) -> Tuple[bool, str]:
    model_value = _model_value_from_key(model_key)
    if not model_value:
        return False, "Неизвестная модель."
    if not user_has_custom_access(user_id):
        return False, "Сначала активируй код."
    async with CUSTOM_PROMPTS_LOCK:
        profile = CUSTOM_PROFILES.setdefault(str(user_id), {})
        profile["model"] = model_value
        profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _save_json_file(CUSTOM_PROMPTS_FILE, CUSTOM_PROFILES)
    return True, f"Модель персонального режима обновлена на {MODEL_LABEL.get(model_key, model_key)}."


def build_custom_status_text(user_id: int) -> str:
    profile = get_custom_profile(user_id)
    if not profile or not profile.get("code"):
        return (
            "🧩 *Персональный режим*\n\n"
            "Активируй одноразовый код, чтобы получить собственный режим с личным системным промптом.\n"
            "После активации код привяжется к твоему аккаунту."
        )

    model_value = profile.get("model") or MODEL_MAP["ds"]
    model_key = _model_key_from_value(model_value) or "ds"
    model_label = MODEL_LABEL.get(model_key, model_key)
    prompt = profile.get("prompt")
    title = profile.get("title")
    status_lines = [
        "🧩 *Персональный режим активирован*",
        f"• Код: `{profile.get('code')}`",
        f"• Модель: {model_label}",
    ]
    if title and prompt:
        status_lines.append(f"• Название: *{title}*")
        snippet = prompt.strip().splitlines()[0][:80]
        status_lines.append(f"• Промпт: {snippet}{'…' if len(prompt) > len(snippet) else ''}")
        status_lines.append("\nМожешь включить режим, обновить текст или удалить его.")
    else:
        status_lines.append("\nКод активирован. Осталось придумать название и текст промпта.")
    return "\n".join(status_lines)




# Sponsor features
SPONSOR_ID = 841874445  # Спонсор)
SPONSOR_MODEL_MAP = {
    "gpt4": "openai/gpt-4",
    "claude": "anthropic/claude-3-sonnet",
    "mixtral": "mistralai/mixtral-8x7b-instruct",
    "deepseek": "deepseek/deepseek-chat-v3-0324"
}
SPONSOR_IMAGE_MODEL = "openai/dall-e-3"

# Массовая генерация изображений - глобальный стиль и модели
BATCH_GLOBAL_STYLE = "Северная магия + реализм, cinematic, ultra detailed, high contrast, слегка более холодный тон, лёгкая туманность/морозный воздух, сияние северного света как магический акцент, PNG, прозрачный фон, no text, no logo."

BATCH_IMAGE_MODELS = {
    "gemini": "google/gemini-2.5-flash-image-preview",  # Бесплатная модель Gemini
    "flux": "black-forest-labs/flux-1-schnell",        # Быстрая модель Flux
    "dalle": "openai/dall-e-3"                        # DALL-E 3 (дорогая, но качественная)
}

BATCH_MODEL_LABELS = {
    "gemini": "🆓 Gemini 2.5 (бесплатная)",
    "flux": "⚡ Flux-1 Schnell (быстрая)",
    "dalle": "🎨 DALL-E 3 (качественная)"
}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("logs.txt", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

START_TIME   = datetime.now()
STATS_FILE   = "stats.json"
CRASH_LOG    = "crash.log"
PENDING_FILE = "pending.json"
CODES_FILE   = "codes.json"
CUSTOM_PROMPTS_FILE = "custom_prompts.json"
CACHE_LIMIT  = 5
PAGE_SIZE    = 6
RECENT_LIMIT = 500

# Функция миграции статистики
def migrate_stats():
    """Миграция старой статистики в новый формат"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                old_data = json.load(f)

            # Если это старый формат (простой словарь без новых ключей)
            if isinstance(old_data, dict) and "private_chats" not in old_data:
                new_stats = {
                    "users": old_data.get("users", []),
                    "chats": old_data.get("chats", []),
                    "private_chats": [],
                    "group_chats": [],
                    "messages": old_data.get("messages", 0),
                    "user_messages": 0,
                    "bot_messages": 0,
                    "image_generations": 0,
                    "active_today": [],
                    "last_reset": datetime.now().strftime("%Y-%m-%d")
                }

                # Сохраняем новую статистику
                with open(STATS_FILE, "w", encoding="utf-8") as f:
                    json.dump(new_stats, f, ensure_ascii=False, indent=2)

                logger.info("✅ Мигрирована старая статистика в новый формат")
                return new_stats

        except Exception as e:
            logger.error(f"Ошибка миграции статистики: {e}")

    return None

MODEL_MAP = {
    "ds": "deepseek/deepseek-chat-v3-0324",
    "mi": "mistralai/mistral-7b-instruct",
    "oc": "openchat/openchat-3.5",
    "ll": "meta-llama/llama-3.1-8b-instruct",
}
MODEL_LABEL = {
    "ds": "🧠 DeepSeek",
    "mi": "⚡ Mistral 7B",
    "oc": "🌀 OpenChat 3.5",
    "ll": "🦙 Llama 3.1 8B",
}
ALLOWED_MODELS = set(MODEL_MAP.values())

MODE_LABEL = {
    "g": "💀 Грубый",
    "n": "😇 Нормальный",
    "b": "💢 Берсерк",
    "p": "😏 Пошлый",
    "c": "🧩 Персональный",
}
MODE_MAP = {
    "grubiy": "g",
    "normal": "n",
    "berserk": "b",
    "poshliy": "p",
    "custom": "c",
    "personal": "c",
}
MODE_MAP_REV = {v:k for k,v in MODE_MAP.items()}

def load_stats():
    # Сначала пытаемся мигрировать старую статистику
    migrate_stats()

    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Обеспечиваем обратную совместимость со старым форматом
                if isinstance(data, dict):
                    # Добавляем отсутствующие ключи для совместимости
                    default_stats = {
                        "users": [],
                        "chats": [],
                        "private_chats": [],
                        "group_chats": [],
                        "messages": 0,
                        "user_messages": 0,
                        "bot_messages": 0,
                        "image_generations": 0,
                        "active_today": [],
                        "last_reset": datetime.now().strftime("%Y-%m-%d")
                    }

                    # Объединяем с существующими данными
                    for key, value in default_stats.items():
                        if key not in data:
                            data[key] = value

                    return data
        except Exception:
            pass

    # Возвращаем статистику по умолчанию
    return {
        "users": [],
        "chats": [],
        "private_chats": [],
        "group_chats": [],
        "messages": 0,
        "user_messages": 0,
        "bot_messages": 0,
        "image_generations": 0,
        "active_today": [],
        "last_reset": datetime.now().strftime("%Y-%m-%d")
    }

def save_stats(d):
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"save_stats error: {e}")

stats = load_stats()

# ---------- Crash log helpers ----------
def log_crash(err_text: str):
    try:
        with open(CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {err_text}\n")
    except Exception as e:
        logger.error(f"write crash log failed: {e}")

def read_crash_tail(n=50):
    if not os.path.exists(CRASH_LOG):
        return ""
    with open(CRASH_LOG, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[-n:])

# ---------- Pending helpers ----------
def load_pending():
    if os.path.exists(PENDING_FILE):
        try:
            with open(PENDING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_pending_list(lst):
    try:
        with open(PENDING_FILE, "w", encoding="utf-8") as f:
            json.dump(lst, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"save pending failed: {e}")

def add_pending(chat_id: int, user_id: int, text: str, waiting_id: int):
    lst = load_pending()
    lst.append({"chat": chat_id, "user": user_id, "text": text, "waiting": waiting_id, "ts": datetime.now().isoformat()})
    save_pending_list(lst)

def remove_pending(chat_id: int, waiting_id: int):
    lst = load_pending()
    lst2 = [x for x in lst if not (x.get("chat")==chat_id and x.get("waiting")==waiting_id)]
    if len(lst2) != len(lst):
        save_pending_list(lst2)

# ---------- Menus ----------
def user_reply_menu():
    return ReplyKeyboardMarkup(
        [[KeyboardButton("📋 Меню")],
         [KeyboardButton("🎨 Генерация изображений")],
         [KeyboardButton("🧹 Обнулить диалог"), KeyboardButton("ℹ️ Инфо")]],
        resize_keyboard=True
    )

def user_home_inline(bot_username: Optional[str] = None, user_id: Optional[int] = None):
    rows = [
        [InlineKeyboardButton("🎭 Выбрать режим", callback_data="u:mode"),
         InlineKeyboardButton("🤖 Выбрать модель", callback_data="u:model")],
        [InlineKeyboardButton("🎨 Генерация изображений", callback_data="u:draw")],
        [InlineKeyboardButton("📩 Поддержка", callback_data="u:sup")]
    ]
    rows.insert(1, [InlineKeyboardButton("🧩 Мой режим", callback_data="u:custom")])

    # Добавляем кнопку спонсора если это он
    if user_id == SPONSOR_ID:
        rows.append([InlineKeyboardButton("👑 Спонсорский доступ", callback_data="sp:home")])

    if bot_username:
        rows.append([InlineKeyboardButton("➕ Добавить в группу", url=f"https://t.me/{bot_username}?startgroup=true")])
    return InlineKeyboardMarkup(rows)

def sponsor_home_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 Приоритетный доступ", callback_data="sp:priority")],
        [InlineKeyboardButton("🎨 Эксклюзивные модели", callback_data="sp:models")],
        [InlineKeyboardButton("⚡ Быстрая генерация", callback_data="sp:fast")],
        [InlineKeyboardButton("📊 Статистика использования", callback_data="sp:stats")],
        [InlineKeyboardButton("💫 Спонсорский режим", callback_data="sp:mode")],
        [InlineKeyboardButton("⬅️ Основное меню", callback_data="u:home")]
    ])

def sponsor_models_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧠 DeepSeek-V3", callback_data="sp:model:deepseek")],
        [InlineKeyboardButton("🚀 GPT-4", callback_data="sp:model:gpt4")],
        [InlineKeyboardButton("⚡ Claude-3", callback_data="sp:model:claude")],
        [InlineKeyboardButton("🎯 Mixtral 8x7B", callback_data="sp:model:mixtral")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="sp:home")]
    ])

def sponsor_modes_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💎 Премиум-режим", callback_data="sp:mode:premium")],
        [InlineKeyboardButton("🚀 Turbo-режим", callback_data="sp:mode:turbo")],
        [InlineKeyboardButton("🎯 Точный режим", callback_data="sp:mode:precise")],
        [InlineKeyboardButton("✨ Творческий режим", callback_data="sp:mode:creative")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="sp:home")]
    ])

def draw_style_inline(scope: str, current: Optional[str] = None):
    """Расширенное меню стилей генерации изображений"""
    styles = [
        ("normal", "🖼 Обычный"),
        ("flirt", "💋 Пикантный"),
        ("land", "🌆 Пейзаж"),
        ("fantasy", "👾 Фэнтези"),
        ("anime", "🎌 Аниме"),
        ("cyber", "🌃 Киберпанк"),
        ("minimal", "⬜ Минимализм"),
        ("realistic", "📷 Реализм"),
    ]
    
    if scope == "admin_batch":
        rows = []
        row = []
        for style_key, style_label in styles:
            row.append(InlineKeyboardButton(style_label, callback_data=f"d:{scope}:s:{style_key}"))
            if len(row) == 2:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:batch_gen")])
    else:
        rows = []
        row = []
        for style_key, style_label in styles:
            check = "✅ " if current == style_key else ""
            row.append(InlineKeyboardButton(f"{check}{style_label}", callback_data=f"d:{scope}:s:{style_key}"))
            if len(row) == 2:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data=("u:home" if scope=="u" else "g:home"))])
    return InlineKeyboardMarkup(rows)

def user_mode_inline(current: Optional[str] = None, has_custom: bool = False):
    cur_short = MODE_MAP.get(current, current)
    rows = [
        [InlineKeyboardButton(("✅ " if cur_short=="g" else "")+MODE_LABEL["g"], callback_data="u:m:g"),
         InlineKeyboardButton(("✅ " if cur_short=="n" else "")+MODE_LABEL["n"], callback_data="u:m:n")],
        [InlineKeyboardButton(("✅ " if cur_short=="b" else "")+MODE_LABEL["b"], callback_data="u:m:b"),
         InlineKeyboardButton(("✅ " if cur_short=="p" else "")+MODE_LABEL["p"], callback_data="u:m:p")]
    ]
    if has_custom:
        rows.append([InlineKeyboardButton(("✅ " if cur_short=="c" else "")+MODE_LABEL["c"], callback_data="u:m:c")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="u:home")])
    return InlineKeyboardMarkup(rows)

def user_model_inline(current: Optional[str] = None):
    cur_short = None
    for k, v in MODEL_MAP.items():
        if v == current:
            cur_short = k
            break
    rows = []
    for k in ("ds","mi","oc","ll"):
        rows.append([InlineKeyboardButton(("✅ " if cur_short==k else "")+MODEL_LABEL[k], callback_data=f"u:md:{k}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="u:home")])
    return InlineKeyboardMarkup(rows)


def user_custom_inline(entry: Optional[Dict[str, Any]]):
    rows = []
    if not entry or not entry.get("code"):
        rows.append([InlineKeyboardButton("🎟 Активировать код", callback_data="u:custom:redeem")])
    else:
        has_prompt = bool(entry.get("prompt"))
        if has_prompt:
            rows.append([InlineKeyboardButton("🚀 Сделать активным", callback_data="u:custom:activate")])
            rows.append([InlineKeyboardButton("✏️ Обновить текст", callback_data="u:custom:edit")])
            rows.append([InlineKeyboardButton("🗑 Удалить режим", callback_data="u:custom:delete")])
        else:
            rows.append([InlineKeyboardButton("🧩 Создать режим", callback_data="u:custom:create")])
        rows.append([InlineKeyboardButton("🤖 Сменить модель", callback_data="u:custom:model")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="u:home")])
    return InlineKeyboardMarkup(rows)


def custom_model_inline(current: Optional[str] = None):
    rows = []
    for key in ("ds", "mi", "oc", "ll"):
        rows.append([
            InlineKeyboardButton(
                ("✅ " if current == key else "") + MODEL_LABEL[key],
                callback_data=f"u:custom:setmodel:{key}"
            )
        ])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="u:custom")])
    return InlineKeyboardMarkup(rows)

def group_home_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎭 Режимы", callback_data="g:mode"),
         InlineKeyboardButton("🤖 Модели", callback_data="g:model")],
        [InlineKeyboardButton("🎨 Арт-режим", callback_data="g:draw")],
        [InlineKeyboardButton("🕹 Когда отвечать", callback_data="g:trg")],
        [InlineKeyboardButton("ℹ️ Состояние", callback_data="g:info")],
        [InlineKeyboardButton("🆘 Репорт админам", callback_data="g:rep")]
    ])

def group_mode_inline(gmode_short: str, user_short: Optional[str], admin: bool, has_custom: bool = False):
    rows = []
    gr = []
    for k in ("g","n","b","p"):
        gr.append(
            InlineKeyboardButton(
                ("✅ " if gmode_short == k else "") + MODE_LABEL[k],
                callback_data=f"g:M:G:{k}",
            )
        )
        if len(gr) == 2:
            rows.append(gr)
            gr = []
    if gr: rows.append(gr)
    if not admin: rows.append([InlineKeyboardButton("🔒 Режим группы — только админ", callback_data="x")])
    rows.append([InlineKeyboardButton("— — Лично мне — —", callback_data="x")])
    personal_keys = ["g", "n", "b", "p"]
    if has_custom:
        personal_keys.append("c")
    pr = []
    for k in personal_keys:
        pr.append(
            InlineKeyboardButton(
                ("✅ " if user_short == k else "") + MODE_LABEL[k],
                callback_data=f"g:M:U:{k}",
            )
        )
        if len(pr) == 2:
            rows.append(pr)
            pr = []
    if pr:
        rows.append(pr)
    rows.append([InlineKeyboardButton("♻️ Сбросить личный режим", callback_data="g:M:U:reset")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="g:home")])
    return InlineKeyboardMarkup(rows)

def group_model_inline(gmodel_key: str, user_key: Optional[str], admin: bool):
    rows = []
    for k in ("ds","mi","oc","ll"):
        rows.append([InlineKeyboardButton(("✅ " if gmodel_key==k else "")+MODEL_LABEL[k],
                                          callback_data=f"g:D:G:{k}")])
    if not admin: rows.append([InlineKeyboardButton("🔒 Модель группы — только админ", callback_data="x")])
    rows.append([InlineKeyboardButton("— — Лично мне — —", callback_data="x")])
    for k in ("ds","mi","oc","ll"):
        rows.append([InlineKeyboardButton(("✅ " if user_key==k else "")+MODEL_LABEL[k],
                                          callback_data=f"g:D:U:{k}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="g:home")])
    return InlineKeyboardMarkup(rows)

def group_triggers_inline(mode: str, mute: bool):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(("✅ " if mode=="all" else "")+"🟢 Всем", callback_data="g:T:all"),
         InlineKeyboardButton(("✅ " if mode=="mention" else "")+"🔔 По упоминанию/словам", callback_data="g:T:men")],
        [InlineKeyboardButton(("✅ " if mode=="allow" else "")+"👤 Только разрешённым", callback_data="g:T:alw")],
        [InlineKeyboardButton(("✅ " if mute else "")+"🔕 Отключить ответы", callback_data="g:T:mt")],
        [InlineKeyboardButton("🔤 Ключевые слова", callback_data="g:T:kw"),
         InlineKeyboardButton("👤 Разрешённые", callback_data="g:T:aw")],
        [InlineKeyboardButton("♻️ Сбросить", callback_data="g:T:rst")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="g:home")]
    ])

def admin_home_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📂 Чаты", callback_data="a:ch")],
        [InlineKeyboardButton("👤 Пользователи", callback_data="a:u")],
        [InlineKeyboardButton("📬 Репорты", callback_data="a:rp")],
        [InlineKeyboardButton("🧾 Логи", callback_data="a:lg"),
         InlineKeyboardButton("📊 Детальная статистика", callback_data="a:st")],
        [InlineKeyboardButton("🎟 Коды режимов", callback_data="a:codes")],
        [InlineKeyboardButton("🎨 Массовая генерация изображений", callback_data="a:batch_gen")],
        [InlineKeyboardButton("🔑 API-ключи", callback_data="a:api")],
        [InlineKeyboardButton("⚠️ Ошибки / Краши", callback_data="a:cr")],
        [InlineKeyboardButton("🚫 Управление банами", callback_data="a:ban")],
        [InlineKeyboardButton("⬅️ Выход", callback_data="a:ex")]
    ])

def admin_detailed_stats_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить статистику", callback_data="a:st:refresh")],
        [InlineKeyboardButton("📅 Сбросить дневную", callback_data="a:st:reset_daily")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:home")]
    ])

def admin_ban_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("👤 Забанить пользователя", callback_data="a:ban:user"),
         InlineKeyboardButton("🔓 Разбанить пользователя", callback_data="a:unban:user")],
        [InlineKeyboardButton("📋 Список пользователей", callback_data="a:ban:users_list"),
         InlineKeyboardButton("📋 Забаненные пользователи", callback_data="a:unban:users_list")],
        [InlineKeyboardButton("👥 Забанить группу", callback_data="a:ban:group"),
         InlineKeyboardButton("🔓 Разбанить группу", callback_data="a:unban:group")],
        [InlineKeyboardButton("🏠 Список групп", callback_data="a:ban:groups_list"),
         InlineKeyboardButton("🏠 Забаненные группы", callback_data="a:unban:groups_list")],
        [InlineKeyboardButton("🚪 Выйти из группы", callback_data="a:ban:leave_group")],
        [InlineKeyboardButton("📋 Общий список банов", callback_data="a:ban:list")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:home")]
    ])


def admin_api_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Обновить", callback_data="a:api:refresh")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:home")]
    ])


def admin_codes_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Сгенерировать код", callback_data="a:codes:new")],
        [InlineKeyboardButton("📋 Активные", callback_data="a:codes:list:active"),
         InlineKeyboardButton("✅ Использованные", callback_data="a:codes:list:used")],
        [InlineKeyboardButton("🚫 Отозванные", callback_data="a:codes:list:revoked")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:home")]
    ])


def admin_batch_gen_inline(current_model="gemini"):
    """Меню массовой генерации изображений"""
    model_label = BATCH_MODEL_LABELS.get(current_model, "Неизвестная модель")
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📝 Ввести список объектов", callback_data="a:batch:input")],
        [InlineKeyboardButton(f"🤖 Модель: {model_label}", callback_data="a:batch:model")],
        [InlineKeyboardButton("🎨 Настройки стиля", callback_data="a:batch:style")],
        [InlineKeyboardButton("💰 Стоимость генерации", callback_data="a:batch:cost")],
        [InlineKeyboardButton("📊 Статистика", callback_data="a:batch:stats")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:home")]
    ])


def admin_batch_model_inline(current_model="gemini"):
    """Меню выбора модели для массовой генерации"""
    rows = []
    for model_key, label in BATCH_MODEL_LABELS.items():
        check = "✅ " if model_key == current_model else ""
        rows.append([InlineKeyboardButton(f"{check}{label}", callback_data=f"a:batch:setmodel:{model_key}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:batch_gen")])
    return InlineKeyboardMarkup(rows)


def admin_batch_style_inline():
    """Меню настроек глобального стиля"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✏️ Изменить глобальный стиль", callback_data="a:batch:edit_style")],
        [InlineKeyboardButton("👀 Посмотреть текущий стиль", callback_data="a:batch:view_style")],
        [InlineKeyboardButton("🔄 Сбросить по умолчанию", callback_data="a:batch:reset_style")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:batch_gen")]
    ])


def admin_batch_cost_inline():
    """Меню стоимости генерации"""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💡 Gemini 2.5: ~0.001$ за изображение", callback_data="x")],
        [InlineKeyboardButton("⚡ Flux-1: ~0.002$ за изображение", callback_data="x")],
        [InlineKeyboardButton("🎨 DALL-E 3: ~0.04$ за изображение", callback_data="x")],
        [InlineKeyboardButton("💰 Пример: 10 изображений Gemini = ~0.01$", callback_data="x")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:batch_gen")]
    ])


def admin_codes_list_inline(items: List[Dict[str, Any]], status: str, page: int = 0):
    rows = []
    total = len(items)
    if total == 0:
        rows.append([InlineKeyboardButton("— нет записей —", callback_data="x")])
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    for entry in items[start:end]:
        code = entry.get("code", "")
        comment = entry.get("comment") or ""
        label = code
        if comment:
            comment_short = comment if len(comment) <= 24 else comment[:21] + "…"
            label += f" • {comment_short}"
        rows.append([InlineKeyboardButton(label, callback_data=f"a:codes:show:{code}:{status}:{page}")])

    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("◀️", callback_data=f"a:codes:list:{status}:{page-1}"))
        nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="x"))
        if end < total:
            nav.append(InlineKeyboardButton("▶️", callback_data=f"a:codes:list:{status}:{page+1}"))
        rows.append(nav)

    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:codes")])
    return InlineKeyboardMarkup(rows)


def admin_code_detail_inline(entry: Dict[str, Any], status: str, page: int):
    rows = []
    code = entry.get("code", "")
    if entry.get("status") == "active":
        rows.append([InlineKeyboardButton("🚫 Отозвать", callback_data=f"a:codes:revoke:{code}:{page}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data=f"a:codes:list:{status}:{page}")])
    return InlineKeyboardMarkup(rows)

def admin_unban_groups_list_inline(groups_data, page=0):
    """Список забаненных групп для разбана"""
    rows = []
    total = len(groups_data)
    if total == 0:
        rows.append([InlineKeyboardButton("— забаненных групп нет —", callback_data="x")])
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")])
        return InlineKeyboardMarkup(rows)

    start = page * PAGE_SIZE
    for idx, group_data in list(enumerate(groups_data))[start:start+PAGE_SIZE]:
        chat_id = group_data["chat_id"]
        title = group_data.get("title", f"Группа {chat_id}")
        rows.append([InlineKeyboardButton(f"🔓 {title}", callback_data=f"a:unbang:{chat_id}")])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️", callback_data=f"a:unbangl:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{(total+PAGE_SIZE-1)//PAGE_SIZE}", callback_data="x"))
    if start + PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("▶️", callback_data=f"a:unbangl:{page+1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")])
    return InlineKeyboardMarkup(rows)

def admin_unban_users_list_inline(users_data, page=0):
    """Список забаненных пользователей для разбана"""
    rows = []
    total = len(users_data)
    if total == 0:
        rows.append([InlineKeyboardButton("— забаненных пользователей нет —", callback_data="x")])
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")])
        return InlineKeyboardMarkup(rows)

    start = page * PAGE_SIZE
    for idx, user_data in list(enumerate(users_data))[start:start+PAGE_SIZE]:
        user_id = user_data["user_id"]
        username = user_data.get("username", "")
        first_name = user_data.get("first_name", "")
        label = f"{first_name} {username}".strip() or f"User {user_id}"
        rows.append([InlineKeyboardButton(f"🔓 {label}", callback_data=f"a:unbanu:{user_id}")])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️", callback_data=f"a:unbanul:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{(total+PAGE_SIZE-1)//PAGE_SIZE}", callback_data="x"))
    if start + PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("▶️", callback_data=f"a:unbanul:{page+1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")])
    return InlineKeyboardMarkup(rows)

def admin_ban_users_list_inline(users_data, page=0):
    """Список пользователей для бана"""
    rows = []
    total = len(users_data)
    if total == 0:
        rows.append([InlineKeyboardButton("— пользователей нет —", callback_data="x")])
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")])
        return InlineKeyboardMarkup(rows)

    start = page * PAGE_SIZE
    for idx, user_data in list(enumerate(users_data))[start:start+PAGE_SIZE]:
        user_id = user_data["user_id"]
        username = user_data.get("username", "")
        first_name = user_data.get("first_name", "")
        label = f"{first_name} {username}".strip() or f"User {user_id}"
        rows.append([InlineKeyboardButton(f"👤 {label}", callback_data=f"a:banu:{user_id}")])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️", callback_data=f"a:banul:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{(total+PAGE_SIZE-1)//PAGE_SIZE}", callback_data="x"))
    if start + PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("▶️", callback_data=f"a:banul:{page+1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")])
    return InlineKeyboardMarkup(rows)

def admin_ban_groups_list_inline(groups_data, page=0, action="ban"):
    """Список групп для бана или выхода"""
    rows = []
    total = len(groups_data)
    if total == 0:
        rows.append([InlineKeyboardButton("— групп нет —", callback_data="x")])
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")])
        return InlineKeyboardMarkup(rows)

    start = page * PAGE_SIZE
    for idx, group_data in list(enumerate(groups_data))[start:start+PAGE_SIZE]:
        chat_id = group_data["chat_id"]
        title = group_data.get("title", f"Группа {chat_id}")
        if action == "leave":
            callback_data = f"a:banlg:{chat_id}"
        else:
            callback_data = f"a:bang:{chat_id}"
        rows.append([InlineKeyboardButton(f"👥 {title}", callback_data=callback_data)])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️", callback_data=f"a:bangl:{page-1}:{action}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{(total+PAGE_SIZE-1)//PAGE_SIZE}", callback_data="x"))
    if start + PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("▶️", callback_data=f"a:bangl:{page+1}:{action}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")])
    return InlineKeyboardMarkup(rows)

def admin_chats_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("👤 Личные", callback_data="a:ch:p:0")],
        [InlineKeyboardButton("👥 Группы/Каналы", callback_data="a:ch:g:0")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:home")]
    ])

def admin_chats_list_inline(items, page, cat):
    rows = []
    # Исправляем обработку items - теперь ожидаем кортежи (cid, label, last, count)
    for item in items:
        if isinstance(item, tuple) and len(item) >= 4:
            cid, label, last, count = item
            rows.append([InlineKeyboardButton(f"{label} • {count} msgs", callback_data=f"a:ct:{cid}:{cat}")])
        elif isinstance(item, dict):
            # Альтернативный формат - словарь
            cid = item.get('cid', item.get('chat_id', 'unknown'))
            label = item.get('label', 'Unknown')
            count = item.get('count', 0)
            rows.append([InlineKeyboardButton(f"{label} • {count} msgs", callback_data=f"a:ct:{cid}:{cat}")])
        else:
            # Если непонятный формат, пропускаем
            continue

    if not rows:
        rows.append([InlineKeyboardButton("— нет чатов —", callback_data="x")])

    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️", callback_data=f"a:ch:{cat}:{page-1}"))
    nav.append(InlineKeyboardButton(f"Стр {page+1}", callback_data="x"))
    if len(items) >= PAGE_SIZE:  # Упрощенная проверка на наличие следующей страницы
        nav.append(InlineKeyboardButton("▶️", callback_data=f"a:ch:{cat}:{page+1}"))
    if nav:
        rows.append(nav)

    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:ch")])
    return InlineKeyboardMarkup(rows)


def admin_view_chat_inline(cid, cat):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("10", callback_data=f"a:v:{cid}:10:{cat}"),
         InlineKeyboardButton("25", callback_data=f"a:v:{cid}:25:{cat}"),
         InlineKeyboardButton("50", callback_data=f"a:v:{cid}:50:{cat}")],
        [InlineKeyboardButton("📤 Экспорт", callback_data=f"a:expt:{cid}:{cat}"),
         InlineKeyboardButton("🛠 Настройки", callback_data=f"a:cfg:{cid}:{cat}")],
        [InlineKeyboardButton("⬅️ Назад", callback_data=f"a:ch:{cat}:0")]
    ])


def admin_users_list_inline(users_data, page=0):
    rows = []
    total = len(users_data)
    if total == 0:
        rows.append([InlineKeyboardButton("— пользователей нет —", callback_data="x")])
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:home")])
        return InlineKeyboardMarkup(rows)

    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    for item in users_data[start:end]:
        rows.append([InlineKeyboardButton(item["label"], callback_data=f"a:uv:{item['user_id']}:{page}")])

    total_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀️", callback_data=f"a:u:p:{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{total_pages}", callback_data="x"))
    if end < total:
        nav.append(InlineKeyboardButton("▶️", callback_data=f"a:u:p:{page+1}"))
    if nav:
        rows.append(nav)

    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:home")])
    return InlineKeyboardMarkup(rows)


def admin_user_detail_inline(user_id, page):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🚫 Забанить", callback_data=f"a:banu:{user_id}"),
            InlineKeyboardButton("🔓 Разбанить", callback_data=f"a:unbanu:{user_id}")
        ],
        [InlineKeyboardButton("⬅️ К пользователям", callback_data=f"a:u:p:{page}")]
    ])


def admin_group_cfg_inline():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🎭 Режим группы", callback_data="a:gm")],
        [InlineKeyboardButton("🤖 Модель группы", callback_data="a:gD")],
        [InlineKeyboardButton("🕹 Когда отвечать", callback_data="a:gT")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:back")]
    ])

def admin_group_mode_inline(gshort):
    rows = []
    for k in ("g","n","b","p"):
        rows.append([InlineKeyboardButton(("✅ " if gshort==k else "")+MODE_LABEL[k], callback_data=f"a:gm:{k}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:cfgb")])
    return InlineKeyboardMarkup(rows)

def admin_group_model_inline(gkey):
    rows = []
    for k in ("ds","mi","oc","ll"):
        rows.append([InlineKeyboardButton(("✅ " if gkey==k else "")+MODEL_LABEL[k], callback_data=f"a:gD:{k}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:cfgb")])
    return InlineKeyboardMarkup(rows)

def admin_group_triggers_inline(mode, mute):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(("✅ " if mode=="all" else "")+"🟢 Всем", callback_data="a:gT:all"),
         InlineKeyboardButton(("✅ " if mode=="mention" else "")+"🔔 По упоминанию/словам", callback_data="a:gT:men")],
        [InlineKeyboardButton(("✅ " if mode=="allow" else "")+"👤 Только allowlist", callback_data="a:gT:alw")],
        [InlineKeyboardButton(("✅ " if mute else "")+"🔕 Отключить ответы", callback_data="a:gT:mt")],
        [InlineKeyboardButton("🔤 Ключевые", callback_data="a:gT:kw"),
         InlineKeyboardButton("👤 Allowlist", callback_data="a:gT:aw")],
        [InlineKeyboardButton("♻️ Сбросить", callback_data="a:gT:rst")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="a:cfgb")]
    ])

def admin_reports_inline(reps, page=0):
    rows = []
    total = len(reps)
    if total == 0:
        rows.append([InlineKeyboardButton("— репортов пока нет —", callback_data="x")])
        rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="a:home")])
        return InlineKeyboardMarkup(rows)
    start = page*PAGE_SIZE
    for idx, rep in list(enumerate(reps))[start:start+PAGE_SIZE]:
        who = rep.get("who", f"id {rep.get('user_id')}")
        rows.append([InlineKeyboardButton(f"{who}: {rep.get('text')[:28]}…", callback_data=f"a:rp:{idx}")])
    nav = []
    if page>0: nav.append(InlineKeyboardButton("◀️", callback_data=f"a:rp:p{page-1}"))
    nav.append(InlineKeyboardButton(f"{page+1}/{(total+PAGE_SIZE-1)//PAGE_SIZE}", callback_data="x"))
    if start+PAGE_SIZE < total:
        nav.append(InlineKeyboardButton("▶️", callback_data=f"a:rp:p{page+1}"))
    rows.append(nav)
    rows.append([InlineKeyboardButton("🧹 Очистить все", callback_data="a:rp:cl"),
                 InlineKeyboardButton("⬅️ Назад", callback_data="a:home")])
    return InlineKeyboardMarkup(rows)

def admin_report_view_inline(idx):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✉️ Ответить", callback_data=f"a:rp:r:{idx}")],
        [InlineKeyboardButton("🗑 Удалить", callback_data=f"a:rp:d:{idx}")],
        [InlineKeyboardButton("⬅️ К списку", callback_data="a:rp")]
    ])

# ---------- Config storage ----------
def get_chat_cfg(bot_data: Dict[str,Any], chat_id: int) -> Dict[str,Any]:
    store = bot_data.setdefault("chat_cfg", {})
    return store.setdefault(chat_id, {
        "group_mode": "normal",
        "user_modes": {},
        "group_model": DEFAULT_MODEL,
        "user_models": {},
        "triggers": {"mode":"mention","keywords":[],"allow":[],"mute":False},
        # ДОБАВЛЯЕМ НАСТРОЙКИ БАНА
        "banned_users": [],  # забаненные пользователи
        "is_banned": False,  # забанена ли вся группа
        "left_groups": []    # группы из которых бот вышел
    })
def chat_label(chat, user=None):
    if chat.type == "private":
        name = (user.first_name or "") + ((" " + user.last_name) if user and user.last_name else "")
        uname = f" @{user.username}" if user and user.username else ""
        return f"👤 {name}{uname}".strip() or "👤 (без имени)"
    else:
        title = chat.title or "(без названия)"
        icon = "👥" if chat.type in ("group", "supergroup") else "📣"
        return f"{icon} {title}"
def ensure_store(bot_data, chat_id, chat_type, label):
    info = bot_data.setdefault("chats_info", {})
    info.setdefault(chat_id, {"type": chat_type, "label": label, "last": "", "count": 0})
    bot_data.setdefault("recent", {}).setdefault(chat_id, [])
    bot_data.setdefault("reports", [])
    # Добавляем хранилище для всех сообщений
    bot_data.setdefault("all_messages", {}).setdefault(chat_id, [])

# ---------- Utilities ----------
async def safe_edit_text(query, text, inline_markup=None):
    try:
        await query.message.edit_text(text, reply_markup=inline_markup)
    except BadRequest as e:
        s = str(e)
        if "Message is not modified" in s:
            return
        logger.error(f"safe_edit error: {e}")

async def is_group_admin(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int) -> bool:
    try:
        m = await context.bot.get_chat_member(chat_id, user_id)
        if m.status in ("administrator","creator"): return True
    except Exception as e:
        logger.warning(f"get_chat_member failed: {e}")
    try:
        admins = await context.bot.get_chat_administrators(chat_id)
        return any(a.user.id == user_id for a in admins)
    except Exception as e:
        logger.warning(f"get_chat_administrators failed: {e}")
    return False

def resolve_effective_mode(context, chat_id, user_id, is_private):
    cfg = get_chat_cfg(context.bot_data, chat_id)
    if is_private:
        mode = context.user_data.get("mode")
        if mode == "custom" and not user_has_custom_prompt(user_id):
            context.user_data["mode"] = "normal"
            mode = "normal"
        return mode
    user_modes = cfg.get("user_modes", {})
    mode = user_modes.get(user_id)
    if mode == "custom" and not user_has_custom_prompt(user_id):
        user_modes.pop(user_id, None)
        mode = None
    return mode or cfg.get("group_mode", "normal")

def resolve_effective_model(context, chat_id, user_id, is_private):
    cfg = get_chat_cfg(context.bot_data, chat_id)
    profile = get_custom_profile(user_id)
    if is_private:
        mode = context.user_data.get("mode")
        if mode == "custom" and profile and profile.get("model"):
            return profile["model"]
        return context.user_data.get("model") or DEFAULT_MODEL
    user_mode = cfg.get("user_modes", {}).get(user_id)
    if user_mode == "custom" and profile and profile.get("model"):
        return profile["model"]
    return cfg.get("user_models", {}).get(user_id) or cfg.get("group_model", DEFAULT_MODEL)

def should_respond_in_group(msg, context) -> bool:
    chat = msg.chat
    if chat.type not in ("group","supergroup"): return True
    cfg = get_chat_cfg(context.bot_data, chat.id)
    trg = cfg.get("triggers", {"mode":"mention","keywords":[],"allow":[],"mute":False})
    if trg.get("mute"): return False
    mode = trg.get("mode","mention")
    if mode == "all": return True
    if mode == "allow": return msg.from_user.id in set(trg.get("allow", []))
    txt = (msg.text or "").lower()
    botname = (context.bot.username or "").lower()
    if botname and ("@" + botname in txt or botname in txt): return True
    if any(k.lower() in txt for k in trg.get("keywords", [])): return True
    if msg.reply_to_message and msg.reply_to_message.from_user and msg.reply_to_message.from_user.id == context.bot.id: return True
    return False

def system_text_for_mode(mode: Optional[str], user_id: Optional[int] = None) -> str:
    presets = {
        "grubiy": "Отвечай грубо, по-фене, с матами, коротко и дерзко.",
        "berserk": "Отвечай яростно, с агрессией, будто берсерк на кураже. Используй мощные эмоции и сленг.",
        "poshliy": "Отвечай с флиртом и пошлостями. Старайся быть нежным не сразу намекать на секс но желать намёками на это.",
        "normal": "Отвечай спокойно и вежливо, как умный собеседник.",
    }
    if mode == "custom" and user_id is not None:
        profile = get_custom_profile(user_id)
        prompt = (profile or {}).get("prompt")
        if prompt:
            return prompt
    return presets.get(mode or "", presets["normal"])

def add_to_memory(ctx: ContextTypes.DEFAULT_TYPE, role: str, content: str, chat_id: int = None, user_id: int = None):
    # Сохраняем в память для контекста
    ctx.user_data.setdefault("memory", [])
    ctx.user_data["memory"].append({"role": role, "content": content})
    if len(ctx.user_data["memory"]) > 10:
        ctx.user_data["memory"] = ctx.user_data["memory"][-10:]

    # Сохраняем в общую историю сообщений
    if chat_id is not None and user_id is not None:
        message = ChatMessage(role, user_id, chat_id, content)
        if "all_messages" not in ctx.bot_data:
            ctx.bot_data["all_messages"] = {}
        if chat_id not in ctx.bot_data["all_messages"]:
            ctx.bot_data["all_messages"][chat_id] = []

        ctx.bot_data["all_messages"][chat_id].append(message.to_dict())
        # Ограничиваем размер истории
        if len(ctx.bot_data["all_messages"][chat_id]) > 1000:
            ctx.bot_data["all_messages"][chat_id] = ctx.bot_data["all_messages"][chat_id][-1000:]
def clear_memory(ctx: ContextTypes.DEFAULT_TYPE):
    ctx.user_data["memory"] = []


def extract_error_message(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("code")
            if msg:
                return str(msg)
        elif isinstance(err, str):
            return err
        msg = payload.get("message")
        if isinstance(msg, str):
            return msg
    return None


# ========== RATE LIMITING ==========
def check_rate_limit(user_id: int, action_type: str = "message") -> Tuple[bool, Optional[str]]:
    """
    Проверяет rate limit для пользователя.
    Возвращает (разрешено, сообщение_об_ошибке)
    """
    now = time.time()
    
    if user_id not in rate_limit_data:
        rate_limit_data[user_id] = {
            "messages": [],
            "images": [],
            "cooldown_until": 0
        }
    
    user_data = rate_limit_data[user_id]
    
    # Проверяем cooldown
    if user_data["cooldown_until"] > now:
        remaining = int(user_data["cooldown_until"] - now)
        return False, f"⏳ Слишком много запросов! Подожди {remaining} сек."
    
    # Очищаем старые записи
    cutoff = now - RATE_LIMIT_WINDOW
    user_data["messages"] = [t for t in user_data["messages"] if t > cutoff]
    user_data["images"] = [t for t in user_data["images"] if t > cutoff]
    
    if action_type == "message":
        if len(user_data["messages"]) >= RATE_LIMIT_MESSAGES:
            user_data["cooldown_until"] = now + RATE_LIMIT_COOLDOWN
            return False, f"⏳ Превышен лимит сообщений ({RATE_LIMIT_MESSAGES}/мин). Подожди {RATE_LIMIT_COOLDOWN} сек."
        user_data["messages"].append(now)
    
    elif action_type == "image":
        if len(user_data["images"]) >= RATE_LIMIT_IMAGES:
            user_data["cooldown_until"] = now + RATE_LIMIT_COOLDOWN
            return False, f"⏳ Превышен лимит генераций ({RATE_LIMIT_IMAGES}/мин). Подожди {RATE_LIMIT_COOLDOWN} сек."
        user_data["images"].append(now)
    
    return True, None


# ========== RETRY WITH EXPONENTIAL BACKOFF ==========
async def retry_async(func, *args, max_attempts: int = RETRY_MAX_ATTEMPTS, **kwargs):
    """
    Выполняет функцию с повторными попытками при ошибках.
    Использует экспоненциальную задержку.
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError, NetworkError) as e:
            last_exception = e
            if attempt < max_attempts - 1:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                logger.warning(f"Retry {attempt + 1}/{max_attempts} after {delay}s: {type(e).__name__}")
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_attempts} attempts failed: {e}")
    
    raise last_exception


# ========== USER STATS ==========
def load_user_stats() -> Dict[str, Any]:
    """Загружает персональную статистику пользователей"""
    if os.path.exists(USER_STATS_FILE):
        try:
            with open(USER_STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user stats: {e}")
    return {}


def save_user_stats(data: Dict[str, Any]) -> None:
    """Сохраняет персональную статистику"""
    try:
        with open(USER_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving user stats: {e}")


def update_user_stats(user_id: int, action: str, value: int = 1) -> None:
    """
    Обновляет статистику пользователя.
    action: 'messages', 'images', 'likes', 'dislikes'
    """
    user_stats = load_user_stats()
    uid = str(user_id)
    
    if uid not in user_stats:
        user_stats[uid] = {
            "messages": 0,
            "images": 0,
            "likes": 0,
            "dislikes": 0,
            "first_use": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_use": None
        }
    
    user_stats[uid][action] = user_stats[uid].get(action, 0) + value
    user_stats[uid]["last_use"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    save_user_stats(user_stats)


def get_user_stats(user_id: int) -> Dict[str, Any]:
    """Получает статистику пользователя"""
    user_stats = load_user_stats()
    uid = str(user_id)
    
    return user_stats.get(uid, {
        "messages": 0,
        "images": 0,
        "likes": 0,
        "dislikes": 0,
        "first_use": None,
        "last_use": None
    })


# ========== ADMIN NOTIFICATIONS ==========
async def notify_admin(bot, message: str, priority: str = "info") -> None:
    """
    Отправляет уведомление админу.
    priority: 'info', 'warning', 'error'
    """
    if not ADMIN_ID:
        return
    
    icons = {
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "🚨",
        "success": "✅",
        "user": "👤"
    }
    
    icon = icons.get(priority, "📬")
    
    try:
        await bot.send_message(
            chat_id=ADMIN_ID,
            text=f"{icon} {message}",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Failed to notify admin: {e}")


async def notify_new_user(bot, user, total_users: int) -> None:
    """Уведомляет админа о новом пользователе"""
    if not NOTIFY_NEW_USERS or not ADMIN_ID:
        return
    
    # Уведомляем только каждого N-го пользователя или при круглых числах
    if total_users % NOTIFY_NEW_USERS_THRESHOLD == 0 or total_users in [1, 5, 10, 25, 50, 100, 250, 500, 1000]:
        name = user.first_name or "Без имени"
        username = f"@{user.username}" if user.username else "нет username"
        
        await notify_admin(
            bot,
            f"*Новый пользователь #{total_users}!*\n"
            f"👤 {name} ({username})\n"
            f"🆔 `{user.id}`",
            priority="user"
        )


async def notify_error(bot, error_type: str, details: str) -> None:
    """Уведомляет админа о критической ошибке"""
    if not NOTIFY_ERRORS or not ADMIN_ID:
        return
    
    await notify_admin(
        bot,
        f"*Ошибка: {error_type}*\n```\n{details[:500]}\n```",
        priority="error"
    )


# ========== FEEDBACK (RATINGS) ==========
def feedback_inline(message_id: int) -> InlineKeyboardMarkup:
    """Создает кнопки оценки ответа"""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👍", callback_data=f"fb:like:{message_id}"),
            InlineKeyboardButton("👎", callback_data=f"fb:dislike:{message_id}")
        ]
    ])


def record_feedback(message_id: int, user_id: int, rating: str) -> bool:
    """
    Записывает оценку ответа.
    rating: 'like' или 'dislike'
    Возвращает True если оценка записана, False если уже оценено
    """
    key = f"{message_id}:{user_id}"
    
    if key in feedback_data:
        return False  # Уже оценено
    
    feedback_data[key] = {
        "rating": rating,
        "timestamp": datetime.now().isoformat()
    }
    
    # Обновляем статистику пользователя
    update_user_stats(user_id, "likes" if rating == "like" else "dislikes")
    
    return True

# ---------- AI ----------
async def call_ai(ctx, prompt: str, mode: Optional[str], model: Optional[str], user_id: Optional[int] = None, chat_id: Optional[int] = None) -> str:
    if not prompt.strip(): return "⚠️ Пустой запрос. Напиши что-нибудь."
    if not mode: return "⚙️ Сначала выбери режим (в меню)."

    # Используем спонсорскую модель если выбрана
    use_model = model or DEFAULT_MODEL
    if user_id == SPONSOR_ID:
        sponsor_model = ctx.user_data.get("sponsor_model")
        if sponsor_model:
            use_model = sponsor_model

    if use_model not in ALLOWED_MODELS and use_model not in SPONSOR_MODEL_MAP.values():
        use_model = DEFAULT_MODEL

    add_to_memory(ctx, "user", prompt)
    system_prompt = system_text_for_mode(mode, user_id)
    if not system_prompt:
        system_prompt = system_text_for_mode("normal", user_id)
    msgs = [{"role": "system", "content": system_prompt}] + ctx.user_data.get("memory", [])
    if not all(m.get("content") for m in msgs):
        return "⚠️ Ошибка в истории диалога. Нажми «🧹 Обнулить диалог» и попробуй снова."
    async def _post(mid: str, api_key: str):
        payload = {"model": mid, "messages": msgs}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as s:
            async with s.post("https://openrouter.ai/api/v1/chat/completions",
                              json=payload, headers=headers,
                              timeout=aiohttp.ClientTimeout(total=30)) as r:
                return r.status, (await r.json()) if r.content_type == "application/json" else {}

    last_status: Any = None
    last_error_text: Optional[str] = None
    tried_indices: Set[int] = set()

    try:
        attempts = AI_KEY_MANAGER.count
        for _ in range(attempts):
            api_key, key_idx = await AI_KEY_MANAGER.acquire(tried_indices)
            try:
                status, js = await _post(use_model, api_key)
            except Exception as post_err:
                logger.error(f"call_ai post error: {post_err}")
                last_status = "network"
                last_error_text = str(post_err)
                await AI_KEY_MANAGER.record_error(key_idx, "network", last_error_text)
                tried_indices.add(key_idx)
                continue

            last_status = status

            if status == 200:
                await AI_KEY_MANAGER.record_success(key_idx)
                reply = js["choices"][0]["message"]["content"].strip()
                # cache reply so repeated prompts respond faster
                add_to_memory(ctx, "assistant", reply, chat_id, user_id)
                cache = ctx.chat_data.setdefault("cache", {})
                cache[prompt] = reply
                if len(cache) > CACHE_LIMIT: cache.pop(next(iter(cache)))
                return reply

            if status == 402:
                error_text = extract_error_message(js) or "Недостаток средств"
                last_error_text = error_text
                logger.warning("call_ai: API key вернул 402 (недостаточно средств), пробуем следующий.")
                await AI_KEY_MANAGER.record_error(key_idx, status, error_text)
                tried_indices.add(key_idx)
                if len(tried_indices) < AI_KEY_MANAGER.count:
                    continue
                break

            if status in (400,403) and use_model != DEFAULT_MODEL:
                status2, js2 = await _post(DEFAULT_MODEL, api_key)
                if status2 == 200:
                    await AI_KEY_MANAGER.record_success(key_idx)
                    reply = js2["choices"][0]["message"]["content"].strip()
                    add_to_memory(ctx, "assistant", reply, chat_id, user_id)
                    return f"Переключил запрос на модель по умолчанию, потому что выбранная недоступна.\n\n{reply}"
                error_text = extract_error_message(js2)
                await AI_KEY_MANAGER.record_error(key_idx, status2, error_text)
                detail = f" {error_text}" if error_text else " Проверь доступность модели или ключа."
                return f"Ошибка API: {status2}.{detail}"

            error_text = extract_error_message(js)
            last_error_text = error_text or last_error_text
            await AI_KEY_MANAGER.record_error(key_idx, status, error_text)
            detail = f" {error_text}" if error_text else " Проверь параметры запроса или повтори попытку."
            return f"Ошибка API: {status}.{detail}"

        if last_status == 402:
            if AI_KEY_MANAGER.count > 1:
                return "Все доступные API-ключи OpenRouter исчерпаны. Добавь новые ключи или пополни баланс."
            return "Баланс на OpenRouter закончился. Пополни счёт: https://openrouter.ai/account"

        if last_status == "network":
            detail = f" {last_error_text}" if last_error_text else " Проверь подключение к сети и попробуй снова."
            return f"Ошибка сети OpenRouter.{detail}"

        status_text = last_status if last_status is not None else "unknown"
        detail = f" {last_error_text}" if last_error_text else " Проверь параметры или повтори попытку."
        return f"Ошибка API: {status_text}.{detail}"
    except asyncio.TimeoutError:
        return "⏱ Сервер долго отвечает. Попробуй ещё раз."
    except Exception as e:
        logger.error(f"call_ai error: {e}")
        return "Ошибка соединения с моделью."

async def bg_generate(context: ContextTypes.DEFAULT_TYPE, chat_id: int, waiting_id: int, text: str, mode: str, model: str, user_id: int):
    """Фоновая генерация ответа ИИ с кнопками оценки"""
    reply = await call_ai(context, text, mode, model, user_id, chat_id)
    
    # Обновляем персональную статистику
    update_user_stats(user_id, "messages")
    
    try:
        # Отправляем ответ с кнопками оценки
        sent_msg = await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=waiting_id,
            text=reply,
            reply_markup=feedback_inline(waiting_id)
        )
    except BadRequest:
        try:
            sent_msg = await context.bot.send_message(
                chat_id=chat_id,
                text=reply,
                reply_markup=feedback_inline(waiting_id)
            )
        except Exception as e:
            logger.error(f"send_message after edit fail: {e}")
    finally:
        remove_pending(chat_id, waiting_id)

# ---------- Image Generation ----------
def _style_prefix(style: str) -> str:
    """Возвращает префикс промпта для стиля генерации"""
    styles = {
        "flirt": "tasteful romantic glamour portrait, soft lighting, elegant, cinematic; suggestive but safe; no nudity. ",
        "fantasy": "fantasy art, highly detailed, intricate, magical atmosphere, dramatic lighting. ",
        "land": "landscape photography, dramatic sky, volumetric light, ultra-detailed, wide angle. ",
        "anime": "anime style, vibrant colors, detailed linework, Japanese animation aesthetic, studio quality. ",
        "cyber": "cyberpunk aesthetic, neon lights, futuristic cityscape, high tech low life, blade runner style, glowing elements. ",
        "minimal": "minimalist design, clean lines, simple shapes, negative space, modern aesthetic, subtle colors. ",
        "realistic": "photorealistic, hyper-detailed, natural lighting, DSLR quality, 8K resolution, lifelike textures. ",
    }
    return styles.get(style, "")  # normal = пустой префикс

# Словарь названий стилей для отображения
STYLE_NAMES = {
    "normal": "Обычный",
    "flirt": "Пикантный",
    "land": "Пейзаж",
    "fantasy": "Фэнтези",
    "anime": "Аниме",
    "cyber": "Киберпанк",
    "minimal": "Минимализм",
    "realistic": "Реализм",
}

async def generate_image_openrouter(prompt: str, style: str = "normal", aspect_ratio: Optional[str] = None, model: Optional[str] = None, user_id: Optional[int] = None) -> Optional[BytesIO]:
    """Генерирует картинку через OpenRouter"""
    # Используем лучшую модель для спонсора
    if user_id == SPONSOR_ID:
        use_model = SPONSOR_IMAGE_MODEL  # DALL-E 3 для спонсора
    else:
        use_model = model or IMAGE_MODEL  # Gemini для обычных

    full_prompt = f"{_style_prefix(style)}{prompt}".strip()
    payload = {
        "model": use_model,
        "messages": [{"role": "user", "content": full_prompt}],
        "modalities": ["image", "text"]
    }
    if aspect_ratio:
        payload["image_config"] = {"aspect_ratio": aspect_ratio}

    last_status: Any = None
    last_error_text: Optional[str] = None
    tried_indices: Set[int] = set()

    async with aiohttp.ClientSession() as s:
        attempts = AI_KEY_MANAGER.count
        for _ in range(attempts):
            api_key, key_idx = await AI_KEY_MANAGER.acquire(tried_indices)
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            try:
                async with s.post("https://openrouter.ai/api/v1/chat/completions",
                                  json=payload, headers=headers,
                                  timeout=aiohttp.ClientTimeout(total=180)) as r:
                    last_status = r.status
                    if r.status == 402:
                        logger.warning("Image generation hit 402 Payment Required, rotating API key.")
                        error_text = "Недостаток средств для генерации"
                        last_error_text = error_text
                        await AI_KEY_MANAGER.record_error(key_idx, r.status, error_text)
                        tried_indices.add(key_idx)
                        if len(tried_indices) < AI_KEY_MANAGER.count:
                            continue
                        break
                    if r.status != 200:
                        txt = await r.text()
                        last_error_text = txt[:200]
                        await AI_KEY_MANAGER.record_error(key_idx, r.status, last_error_text)
                        logger.error(f"image gen http {r.status}: {last_error_text}")
                        return None
                    js = await r.json()
                    try:
                        msg = js["choices"][0]["message"]
                        imgs = msg.get("images") or []
                        if not imgs:
                            raise ValueError("image gen: no images in response")
                        data_url = imgs[0]["image_url"]["url"]
                        if "," in data_url:
                            b64 = data_url.split(",", 1)[1]
                        else:
                            raise ValueError("image gen: unexpected image url format")
                        raw = base64.b64decode(b64)
                        bio = BytesIO(raw)
                        bio.name = "image.png"
                        bio.seek(0)
                        await AI_KEY_MANAGER.record_success(key_idx)
                        return bio
                    except Exception as e:
                        err_text = str(e)
                        last_error_text = err_text
                        await AI_KEY_MANAGER.record_error(key_idx, "parse", err_text)
                        logger.error(f"image parse error: {e}")
                        return None
            except Exception as post_err:
                last_status = "network"
                last_error_text = str(post_err)
                await AI_KEY_MANAGER.record_error(key_idx, "network", last_error_text)
                logger.error(f"image gen network error: {post_err}")
                tried_indices.add(key_idx)
                if len(tried_indices) < AI_KEY_MANAGER.count:
                    continue

    if last_status == 402:
        logger.error("Image generation failed: все API-ключи OpenRouter вернули 402.")
    elif last_error_text:
        logger.error(f"Image generation failed: {last_error_text}")
    return None

async def bg_generate_image(context: ContextTypes.DEFAULT_TYPE, chat_id: int, waiting_id: int, prompt: str, style: str, aspect_ratio: Optional[str] = None, user_id: Optional[int] = None):
    try:
        img = await generate_image_openrouter(prompt, style=style, aspect_ratio=aspect_ratio, user_id=user_id)
        if img is None:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=waiting_id,
                text="💰 Недостаточно средств на счету OpenRouter для генерации изображений.\n\nПополни баланс на https://openrouter.ai/account"
            )
            return

        if img:
            # УВЕЛИЧИВАЕМ СЧЕТЧИК ГЕНЕРАЦИЙ
            stats["image_generations"] = stats.get("image_generations", 0) + 1
            save_stats(stats)
            
            # Обновляем персональную статистику
            if user_id:
                update_user_stats(user_id, "images")

            caption = f"🎨 Стиль: {STYLE_NAMES.get(style, 'Обычный')}"
            if user_id == SPONSOR_ID:
                caption += " | 👑 DALL-E 3"
            await context.bot.edit_message_text(chat_id=chat_id, message_id=waiting_id, text="✅ Готово! Отправляю изображение...")
            await context.bot.send_photo(chat_id=chat_id, photo=img, caption=caption)
        else:
            await context.bot.edit_message_text(chat_id=chat_id, message_id=waiting_id, text="⚠️ Не удалось сгенерировать изображение. Попробуй другой промпт или стиль.")
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        await context.bot.edit_message_text(chat_id=chat_id, message_id=waiting_id, text="❌ Ошибка генерации изображения")
    finally:
        remove_pending(chat_id, waiting_id)

async def generate_single_image_async(prompt: str, style: str = "normal", aspect_ratio: Optional[str] = None, user_id: Optional[int] = None) -> Optional[BytesIO]:
    """Асинхронная генерация одного изображения"""
    return await generate_image_openrouter(prompt, style, aspect_ratio, None, user_id)

async def bg_batch_generate(context: ContextTypes.DEFAULT_TYPE, chat_id: int, objects_list: List[Dict[str, str]], style: str, user_id: int):
    """Фоновая массовая генерация изображений"""
    total = len(objects_list)
    success_count = 0
    error_count = 0
    
    for i, obj in enumerate(objects_list, 1):
        try:
            # Отправляем прогресс
            progress_msg = f"🎨 Генерация {i}/{total}: {obj['name']}"
            await context.bot.send_message(chat_id=chat_id, text=progress_msg)
            
            # Генерируем изображение
            prompt = obj['description']
            aspect_ratio = obj.get('ratio', '16:9')
            
            img = await generate_single_image_async(prompt, style, aspect_ratio, user_id)
            
            if img:
                # Увеличиваем счетчик успешных генераций
                success_count += 1
                
                # Отправляем изображение с подписью
                caption = f"✅ {obj['name']} ({aspect_ratio})"
                await context.bot.send_photo(chat_id=chat_id, photo=img, caption=caption)
            else:
                error_count += 1
                await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка генерации: {obj['name']}")
                
        except Exception as e:
            error_count += 1
            logger.error(f"Ошибка при генерации {obj['name']}: {e}")
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Критическая ошибка: {obj['name']} - {str(e)[:100]}")
    
    # Отправляем итоговую статистику
    final_msg = f"""
🎯 **Массовая генерация завершена!**

📊 Результаты:
• Всего объектов: {total}
• Успешно: ✅ {success_count}
• Ошибок: ❌ {error_count}

⚡ Эффективность: {(success_count/total*100):.1f}%
    """
    
    try:
        await context.bot.send_message(chat_id=chat_id, text=final_msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка отправки финального сообщения: {e}")

# ---------- Commands ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user, chat = update.effective_user, update.effective_chat
    ensure_store(context.bot_data, chat.id, chat.type, chat_label(chat, user))

    if chat.type == "private":
        # Специальное приветствие для спонсора
        if user.id == SPONSOR_ID:
            welcome = (
                f"👑 Добро пожаловать, Спонсор! 💝\n\n"
                f"Спасибо за поддержку проекта! Ваш вклад помогает развитию бота.\n\n"
                f"Вам доступны эксклюзивные функции через команду /sponsor\n\n"
                f"Приятного использования! 🚀"
            )
        else:
            welcome = (
                f"Привет, {user.first_name or 'друг'}! 👋\n\n"
                f"Я — DeepSeek 🤖, твой AI-напарник ({VERSION})\n\n"
                f"Можешь общаться со мной в разных режимах и генерировать изображения!\n\n"
                f"*Начни общение прямо сейчас!* 🚀"
            )

        await update.message.reply_text(welcome, reply_markup=user_reply_menu())
        await update.message.reply_text("Главное меню:", reply_markup=user_home_inline(context.bot.username, user.id))
    else:
        await update.message.reply_text(
            f"✅ Бот подключён к группе ({VERSION}).\n"
            "Администраторы настраивают поведение для всей группы.\n"
            "Каждый участник может выбрать личный режим и модель.\n"
            "Для генерации изображения: команда /draw или «🎨 Арт-режим».",
            reply_markup=group_home_inline()
        )

async def groupmenu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat, user = update.effective_chat, update.effective_user
    if chat.type not in ("group","supergroup"):
        await update.message.reply_text("Эта команда только для групп.")
        return
    ensure_store(context.bot_data, chat.id, chat.type, chat_label(chat, user))
    await update.message.reply_text("Меню группы:", reply_markup=group_home_inline())

async def admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Нет доступа.")
        return
    await update.message.reply_text(f"👑 Панель управления ({VERSION}):", reply_markup=admin_home_inline())

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help — показывает справку по командам"""
    user = update.effective_user
    is_admin = user.id == ADMIN_ID
    is_sponsor = user.id == SPONSOR_ID
    
    help_text = (
        f"📚 **Справка по командам** ({VERSION})\n\n"
        "🔹 **Основные команды:**\n"
        "/start — Запуск бота и главное меню\n"
        "/help — Эта справка\n"
        "/draw <описание> — Генерация изображения\n"
        "/clear — Очистить историю диалога\n"
        "/mystats — Твоя персональная статистика\n\n"
        "🔹 **В группах:**\n"
        "/groupmenu — Меню настроек группы\n\n"
        "🎨 **Стили изображений:**\n"
        "Обычный, Пикантный, Пейзаж, Фэнтези,\n"
        "Аниме, Киберпанк, Минимализм, Реализм\n\n"
        "💡 **Подсказки:**\n"
        "• После ответа бота можно оценить его 👍/👎\n"
        "• Лимит: 30 сообщений и 10 картинок в минуту\n"
    )
    
    if is_sponsor:
        help_text += (
            "\n👑 **Спонсорские команды:**\n"
            "/sponsor — Спонсорская панель\n"
            "• Без лимитов на запросы!\n"
        )
    
    if is_admin:
        help_text += (
            "\n🛠 **Админ-команды:**\n"
            "/admin — Панель управления\n"
            "/ban <id> — Забанить пользователя\n"
            "/unban <id> — Разбанить пользователя\n"
            "/bangroup — Забанить группу и выйти\n"
            "/stats — Быстрая статистика\n"
        )
    
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /clear — очистить историю диалога"""
    context.user_data["memory"] = []
    await update.message.reply_text("🧹 История диалога очищена!", reply_markup=user_reply_menu())

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /stats — быстрая статистика (только для админа)"""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Нет доступа.")
        return
    
    uptime = datetime.now() - START_TIME
    hours, rem = divmod(int(uptime.total_seconds()), 3600)
    minutes, _ = divmod(rem, 60)
    
    text = (
        f"📊 **Быстрая статистика**\n\n"
        f"👥 Пользователей: {len(stats.get('users', []))}\n"
        f"💬 Сообщений: {stats.get('messages', 0)}\n"
        f"🎨 Изображений: {stats.get('image_generations', 0)}\n"
        f"📅 Активных сегодня: {len(stats.get('active_today', []))}\n"
        f"⏱ Аптайм: {hours}ч {minutes}м\n"
        f"🔖 Версия: {VERSION}"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def mystats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /mystats — персональная статистика пользователя"""
    user = update.effective_user
    user_stat = get_user_stats(user.id)
    
    # Форматируем даты
    first_use = user_stat.get("first_use", "Неизвестно")
    last_use = user_stat.get("last_use", "Неизвестно")
    
    # Считаем рейтинг
    likes = user_stat.get("likes", 0)
    dislikes = user_stat.get("dislikes", 0)
    total_feedback = likes + dislikes
    
    if total_feedback > 0:
        satisfaction = (likes / total_feedback) * 100
        rating_bar = "🟢" * int(satisfaction / 20) + "⚪" * (5 - int(satisfaction / 20))
    else:
        satisfaction = 0
        rating_bar = "⚪⚪⚪⚪⚪"
    
    text = (
        f"📊 **Твоя статистика**\n\n"
        f"👤 {user.first_name or 'Пользователь'}\n"
        f"🆔 `{user.id}`\n\n"
        f"💬 Сообщений отправлено: {user_stat.get('messages', 0)}\n"
        f"🎨 Изображений создано: {user_stat.get('images', 0)}\n\n"
        f"**Оценки ответов бота:**\n"
        f"👍 Понравилось: {likes}\n"
        f"👎 Не понравилось: {dislikes}\n"
        f"📈 Удовлетворённость: {rating_bar} {satisfaction:.0f}%\n\n"
        f"📅 Первое использование: {first_use}\n"
        f"🕐 Последняя активность: {last_use}"
    )
    
    await update.message.reply_text(text, parse_mode="Markdown")

async def sponsor_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != SPONSOR_ID:
        await update.message.reply_text("⛔ Эта команда только для спонсора.")
        return

    await update.message.reply_text(
        "👑 **Спонсорская панель**\n\n"
        "Благодарим за поддержку проекта! 💝\n"
        "Вам доступны эксклюзивные возможности:\n\n"
        "• 🚀 Приоритетная обработка запросов\n"
        "• 🎨 Премиум-модели (GPT-4, Claude-3, Mixtral)\n"
        "• ⚡ DALL-E 3 для генерации изображений\n"
        "• 💫 Специальные режимы общения\n"
        "• 📊 Расширенная статистика",
        reply_markup=sponsor_home_inline()
    )

async def draw_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /draw: /draw <описание>. Если без описания — попросим выбрать стиль и ввести описание."""
    user, chat = update.effective_user, update.effective_chat
    ensure_store(context.bot_data, chat.id, chat.type, chat_label(chat, user))
    prompt = " ".join(context.args or [])
    if prompt:
        style = context.user_data.get("last_draw_style", "normal")
        waiting = await update.message.reply_text("🎨 Генерирую изображение... ⌛")
        add_pending(chat.id, user.id, f"[DRAW] {prompt}", waiting.message_id)
        asyncio.create_task(bg_generate_image(context, chat.id, waiting.message_id, prompt, style, user_id=user.id))
        return
    # без промпта — показать меню выбора стиля
    if chat.type == "private":
        await update.message.reply_text("Выбери стиль генерации:", reply_markup=draw_style_inline("u", context.user_data.get("last_draw_style")))
    else:
        await update.message.reply_text("Выбери стиль генерации:", reply_markup=draw_style_inline("g", context.user_data.get("last_draw_style")))

# ---------- Callbacks ----------
async def cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    data = q.data or ""
    user, chat = update.effective_user, update.effective_chat
    
    # ========== ОБРАБОТКА ОЦЕНОК (FEEDBACK) ==========
    if data.startswith("fb:"):
        parts = data.split(":")
        if len(parts) >= 3:
            action = parts[1]  # like или dislike
            msg_id = parts[2]
            
            if action in ("like", "dislike"):
                success = record_feedback(int(msg_id), user.id, action)
                
                if success:
                    emoji = "👍" if action == "like" else "👎"
                    await q.answer(f"{emoji} Спасибо за оценку!", show_alert=False)
                    
                    # Убираем кнопки после оценки
                    try:
                        await q.message.edit_reply_markup(reply_markup=None)
                    except Exception:
                        pass
                else:
                    await q.answer("Ты уже оценил этот ответ!", show_alert=True)
            return
    
    await q.answer()
    ensure_store(context.bot_data, chat.id, chat.type, chat_label(chat, user))

    async def send_admin_stats() -> None:
        # Ensure all expected keys exist
        for key in ["private_chats", "group_chats", "user_messages", "bot_messages", "image_generations", "active_today"]:
            if key not in stats:
                stats[key] = [] if key.endswith("s") else 0

        if "last_reset" not in stats:
            stats["last_reset"] = datetime.now().strftime("%Y-%m-%d")

        chats_info = context.bot_data.get("chats_info", {})

        private_chats_count = sum(1 for info in chats_info.values() if info["type"] == "private")
        group_chats_count = sum(1 for info in chats_info.values() if info["type"] in ("group", "supergroup"))

        active_chats_count = 0
        for info in chats_info.values():
            last_seen = info.get("last")
            if not last_seen:
                continue
            try:
                last_date = datetime.strptime(last_seen, "%Y-%m-%d %H:%M")
            except ValueError:
                continue
            if (datetime.now() - last_date).days <= 30:
                active_chats_count += 1

        all_messages = context.bot_data.get("all_messages", {})
        unique_users = {msg["user_id"] for messages in all_messages.values() for msg in messages if msg.get("user_id")}

        stats["private_chats"] = list(
            {cid for cid, info in chats_info.items() if info["type"] == "private"}.union(stats.get("private_chats", []))
        )
        stats["group_chats"] = list(
            {cid for cid, info in chats_info.items() if info["type"] in ("group", "supergroup")}.union(stats.get("group_chats", []))
        )

        bot_msgs = 0
        user_msgs = 0
        for messages in all_messages.values():
            for msg in messages:
                if msg.get("role") == "assistant":
                    bot_msgs += 1
                elif msg.get("role") == "user":
                    user_msgs += 1

        stats["user_messages"] = user_msgs
        stats["bot_messages"] = bot_msgs

        uptime = datetime.now() - START_TIME
        hours, rem = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(rem, 60)

        today = datetime.now().strftime("%Y-%m-%d")
        if stats.get("last_reset") != today:
            stats["active_today"] = []
            stats["last_reset"] = today

        today_users = {
            cid
            for cid, info in chats_info.items()
            if info["type"] == "private" and info.get("last", "").startswith(today)
        }
        stats["active_today"] = list(today_users)

        save_stats(stats)

        text = (
            f"📊 *Детальная статистика бота*\n\n"
            f"👥 *Пользователи:*\n"
            f"• Всего пользователей: {len(stats['users'])}\n"
            f"• Уникальных в истории: {len(unique_users)}\n"
            f"• Активных сегодня: {len(stats['active_today'])}\n\n"
            f"💬 *Чаты:*\n"
            f"• Всего чатов: {len(stats['chats'])}\n"
            f"• Личных чатов: {private_chats_count}\n"
            f"• Групп/каналов: {group_chats_count}\n"
            f"• Активных (30 дней): {active_chats_count}\n\n"
            f"📨 *Сообщения:*\n"
            f"• Всего сообщений: {stats['messages']}\n"
            f"• От пользователей: {stats['user_messages']}\n"
            f"• От бота: {stats['bot_messages']}\n"
            f"• Генераций изображений: {stats.get('image_generations', 0)}\n\n"
            f"⏱ *Система:*\n"
            f"• Аптайм: {hours}ч {minutes}м\n"
            f"• Версия: {VERSION}\n"
            f"• Последнее обновление: {datetime.now().strftime('%H:%M:%S')}"
        )

        await safe_edit_text(q, text, admin_detailed_stats_inline())

    def format_user_name(info: Dict[str, Any], user_id: int) -> str:
        parts = []
        first = info.get("first_name")
        last = info.get("last_name")
        if first:
            parts.append(first)
        if last:
            parts.append(last)
        name = " ".join(parts).strip()
        username = info.get("username")
        if not name and username:
            name = f"@{username}"
        if not name:
            name = f"ID {user_id}"
        return name

    def collect_user_items() -> list:
        users_info = context.bot_data.get("users_info", {})
        items = []
        for uid, info in users_info.items():
            name = format_user_name(info, uid)
            messages = info.get("messages", 0)
            last_activity = info.get("last_activity", "")
            label = f"{name} • {messages} msgs"
            if last_activity:
                label += f" • {last_activity}"
            if len(label) > 64:
                label = label[:61] + "…"
            items.append({
                "user_id": uid,
                "label": label,
                "messages": messages,
                "last_activity": last_activity,
            })
        items.sort(key=lambda x: (x["last_activity"], x["messages"]), reverse=True)
        return items

    # User (DM)
    if data == "u:home":
        await safe_edit_text(q, "Главное меню:", user_home_inline(context.bot.username, user.id))
        return
    if data == "u:mode":
        cur = context.user_data.get("mode")
        cur_short = MODE_MAP.get(cur, cur)
        await safe_edit_text(q, "Выбери режим:", user_mode_inline(cur_short, user_has_custom_prompt(user.id)))
        return
    if data.startswith("u:m:"):
        short = data.split(":")[-1]
        if short == "c":
            if not user_has_custom_prompt(user.id):
                await q.answer("Сначала создай персональный режим.", show_alert=True)
                return
            context.user_data["mode"] = "custom"
        else:
            context.user_data["mode"] = MODE_MAP_REV.get(short, "normal")
        await safe_edit_text(q, f"✅ Режим установлен: {context.user_data['mode']}", user_home_inline(context.bot.username, user.id))
        return
    if data == "u:model":
        cur = context.user_data.get("model")
        await safe_edit_text(q, "Выбери модель:", user_model_inline(cur))
        return
    if data.startswith("u:md:"):
        key = data.split(":")[-1]
        if key in MODEL_MAP:
            context.user_data["model"] = MODEL_MAP[key]
            await safe_edit_text(q, "✅ Модель переключена!", user_home_inline(context.bot.username, user.id))
        else:
            await safe_edit_text(q, "❌ Модель недоступна.", user_home_inline(context.bot.username, user.id))
        return
    if data == "u:sup":
        context.user_data["support_mode"] = True
        await safe_edit_text(q, "Опиши проблему одним сообщением — отправлю админу ⛑️", user_home_inline(context.bot.username, user.id))
        return
    if data == "u:draw":
        await safe_edit_text(q, "Выбери стиль генерации:",
                             draw_style_inline("u", context.user_data.get("last_draw_style")))
        return

    if data == "u:custom":
        text = build_custom_status_text(user.id)
        await safe_edit_text(q, text, user_custom_inline(get_custom_profile(user.id)))
        return

    if data == "u:custom:redeem":
        if user_has_custom_access(user.id):
            await q.answer("Код уже активирован.", show_alert=True)
            return
        context.user_data["await_custom_code"] = {"chat_id": chat.id}
        await safe_edit_text(
            q,
            "Введи одноразовый код одним сообщением.\nНапиши «Отмена», чтобы отменить.",
            user_custom_inline(get_custom_profile(user.id))
        )
        return

    if data == "u:custom:create":
        if not user_has_custom_access(user.id):
            await q.answer("Сначала активируй код.", show_alert=True)
            return
        context.user_data["custom_builder"] = {
            "action": "create",
            "chat_id": chat.id,
            "step": "title",
        }
        await safe_edit_text(
            q,
            "Придумай название для режима (1-48 символов) и отправь одним сообщением.\n"
            "Напиши «Отмена», чтобы отменить.",
            user_custom_inline(get_custom_profile(user.id))
        )
        return

    if data == "u:custom:edit":
        if not user_has_custom_prompt(user.id):
            await q.answer("Сначала создай персональный режим.", show_alert=True)
            return
        context.user_data["custom_builder"] = {
            "action": "edit",
            "chat_id": chat.id,
            "step": "title",
        }
        await safe_edit_text(
            q,
            "Отправь новое название режима (или то же самое).\nНапиши «Отмена», чтобы отменить.",
            user_custom_inline(get_custom_profile(user.id))
        )
        return

    if data == "u:custom:delete":
        ok, msg_txt = await delete_custom_prompt(user.id)
        if ok and context.user_data.get("mode") == "custom":
            context.user_data["mode"] = "normal"
        await safe_edit_text(q, msg_txt, user_custom_inline(get_custom_profile(user.id)))
        return

    if data == "u:custom:activate":
        if not user_has_custom_prompt(user.id):
            await q.answer("Сначала создай персональный режим.", show_alert=True)
            return
        context.user_data["mode"] = "custom"
        await safe_edit_text(
            q,
            "Персональный режим активирован. Все следующие ответы будут использовать твой промпт.",
            user_home_inline(context.bot.username, user.id)
        )
        return

    if data == "u:custom:model":
        profile = get_custom_profile(user.id)
        current_key = _model_key_from_value((profile or {}).get("model") or MODEL_MAP["ds"])
        await safe_edit_text(
            q,
            "Выбери модель для персонального режима:",
            custom_model_inline(current_key)
        )
        return

    if data.startswith("u:custom:setmodel:"):
        model_key = data.split(":")[-1]
        ok, msg_txt = await set_custom_model(user.id, model_key)
        await safe_edit_text(
            q,
            msg_txt,
            user_custom_inline(get_custom_profile(user.id))
        )
        return

    # Sponsor system
    if data == "sp:home":
        if user.id != SPONSOR_ID:
            await q.answer("⛔ Только для спонсора", show_alert=True)
            return
        await safe_edit_text(q, "👑 **Спонсорская панель**\n\nВыберите эксклюзивную функцию:", sponsor_home_inline())
        return

    if data.startswith("sp:"):
        if user.id != SPONSOR_ID:
            await q.answer("⛔ Только для спонсора", show_alert=True)
            return

        if data == "sp:models":
            await safe_edit_text(q, "🎨 **Эксклюзивные модели**\n\nДоступны только спонсорам:", sponsor_models_inline())
            return
        elif data == "sp:mode":
            await safe_edit_text(q, "💫 **Специальные режимы**\n\nЭксклюзивные режимы общения:", sponsor_modes_inline())
            return
        elif data == "sp:priority":
            await safe_edit_text(q, "🚀 **Приоритетный доступ активирован!**\n\nВаши запросы обрабатываются в первую очередь.", sponsor_home_inline())
            return
        elif data == "sp:stats":
            stats_text = "📊 **Ваша статистика:**\n\n"
            stats_text += f"• Запросов сегодня: 0\n"
            stats_text += f"• Изображений сегодня: 0\n"
            stats_text += f"• Всего запросов: 0\n"
            stats_text += f"• Использовано DALL-E 3: 0\n\n"
            stats_text += "Спасибо за поддержку! 💝"
            await safe_edit_text(q, stats_text, sponsor_home_inline())
            return
        elif data == "sp:fast":
            await safe_edit_text(q, "⚡ **Ускоренная генерация активирована!**\n\nВаши запросы обрабатываются быстрее.", sponsor_home_inline())
            return

        elif data.startswith("sp:model:"):
            model_key = data.split(":")[-1]
            if model_key in SPONSOR_MODEL_MAP:
                context.user_data["sponsor_model"] = SPONSOR_MODEL_MAP[model_key]
                await safe_edit_text(q, f"✅ Модель установлена: {model_key}", sponsor_models_inline())
            return

        elif data.startswith("sp:mode:"):
            mode_key = data.split(":")[-1]
            context.user_data["sponsor_mode"] = mode_key
            await safe_edit_text(q, f"✅ Режим установлен: {mode_key}", sponsor_modes_inline())
            return

    if data == "g:draw":
        await safe_edit_text(q, "Выбери стиль генерации:",
                             draw_style_inline("g", context.user_data.get("last_draw_style")))
        return

    # Draw style selection (both scopes)
    if data.startswith("d:"):
        parts = data.split(":")
        scope = parts[1]
        if len(parts) >= 4 and parts[2] == "s":
            style = parts[3]
            if style not in ("normal","flirt","land","fantasy"):
                await q.answer("Неверный стиль.")
                return
            
            # Проверяем, не является ли это массовой генерацией для админа
            if scope == "admin_batch" and context.user_data.get("batch_generation"):
                batch_data = context.user_data.pop("batch_generation")
                objects_list = batch_data["objects"]
                
                # Начинаем массовую генерацию
                await safe_edit_text(q, f"🚀 Запускаю массовую генерацию {len(objects_list)} изображений...", admin_batch_gen_inline())
                
                # Запускаем фоновую генерацию
                asyncio.create_task(bg_batch_generate(context, chat.id, objects_list, style, user.id))
                return
            
            context.user_data["last_draw_style"] = style
            context.user_data["draw"] = {"scope": scope, "style": style, "chat": chat.id}
            await safe_edit_text(q, "Ок! Напиши одним сообщением, что нужно нарисовать 🎨", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data=("u:home" if scope=="u" else "g:home"))]]))
            return

    # Group
    if data == "g:home":
        await safe_edit_text(q, "Меню группы:", group_home_inline())
        return
    if data == "g:info":
        cfg = get_chat_cfg(context.bot_data, chat.id)
        trg = cfg.get("triggers", {})
        info = (
            f"🎭 Группа: {context.bot_data.get('chats_info',{}).get(chat.id,{}).get('label','')}\n"
            f"• Режим группы: {cfg.get('group_mode')}\n"
            f"• Модель группы: {cfg.get('group_model')}\n"
            f"• Твой режим: {cfg.get('user_modes',{}).get(user.id,'—')}\n"
            f"• Твоя модель: {cfg.get('user_models',{}).get(user.id,'—')}\n"
            f"• Отклик: {trg.get('mode')} | Слова: {len(trg.get('keywords',[]))} | Allow: {len(trg.get('allow',[]))} | {'🔕' if trg.get('mute') else '🔔'}"
        )
        await safe_edit_text(q, info, group_home_inline())
        return
    if data == "g:rep":
        context.user_data["support_mode"] = True
        await safe_edit_text(q, "Напиши одним сообщением — я отправлю это админу.", group_home_inline())
        return
    if data == "g:mode":
        cfg = get_chat_cfg(context.bot_data, chat.id)
        admin_rights = await is_group_admin(context, chat.id, user.id)
        gshort = MODE_MAP.get(cfg.get("group_mode","normal"), "n")
        us = MODE_MAP.get(cfg.get("user_modes",{}).get(user.id), None)
        has_custom = user_has_custom_prompt(user.id)
        await safe_edit_text(
            q,
            "Режим общения:\n(верх — для всей группы, низ — лично тебе)",
            group_mode_inline(gshort, us, admin_rights, has_custom),
        )
        return
    if data.startswith("g:M:"):
        parts = data.split(":")
        scope = parts[2]
        val = parts[3] if len(parts) > 3 else ""
        cfg = get_chat_cfg(context.bot_data, chat.id)
        admin_rights = await is_group_admin(context, chat.id, user.id)
        if scope=="G":
            if not admin_rights:
                await q.answer("Только админ группы.", show_alert=True)
                return
            cfg["group_mode"] = MODE_MAP_REV.get(val, "normal")
            if cfg["user_modes"].get(user.id) == cfg["group_mode"]:
                cfg["user_modes"].pop(user.id, None)
        else:
            if val=="reset":
                cfg["user_modes"].pop(user.id, None)
            else:
                if val == "c" and not user_has_custom_prompt(user.id):
                    await q.answer("Сначала создай персональный режим.", show_alert=True)
                    return
                cfg["user_modes"][user.id] = MODE_MAP_REV.get(val, "normal")
                if cfg["user_modes"].get(user.id) == cfg["group_mode"]:
                    cfg["user_modes"].pop(user.id, None)
        gshort = MODE_MAP.get(cfg.get("group_mode","normal"),"n")
        us = MODE_MAP.get(cfg.get("user_modes",{}).get(user.id), None)
        await safe_edit_text(q, "✅ Обновлено.", group_mode_inline(gshort, us, admin_rights, user_has_custom_prompt(user.id)))
        return
    if data == "g:model":
        cfg = get_chat_cfg(context.bot_data, chat.id)
        admin_rights = await is_group_admin(context, chat.id, user.id)
        gkey = next((k for k,v in MODEL_MAP.items() if v == cfg.get("group_model")), "ds")
        ukey = next((k for k,v in MODEL_MAP.items() if v == cfg.get("user_models",{}).get(user.id)), None)
        await safe_edit_text(
            q,
            "Модель ИИ:\n(верх — для всей группы, низ — лично тебе)",
            group_model_inline(gkey, ukey, admin_rights),
        )
        return
    if data.startswith("g:D:"):
        _,_,scope,key = data.split(":")
        if key not in MODEL_MAP:
            await q.answer("Нет такой модели.", show_alert=True)
            return
        cfg = get_chat_cfg(context.bot_data, chat.id)
        admin_rights = await is_group_admin(context, chat.id, user.id)
        if scope=="G":
            if not admin_rights:
                await q.answer("Только админ группы.", show_alert=True)
                return
            cfg["group_model"] = MODEL_MAP[key]
        else:
            cfg["user_models"][user.id] = MODEL_MAP[key]
            if cfg["user_models"].get(user.id) == cfg["group_model"]:
                cfg["user_models"].pop(user.id, None)
        gkey = next((k for k,v in MODEL_MAP.items() if v == cfg.get("group_model")), "ds")
        ukey = next((k for k,v in MODEL_MAP.items() if v == cfg.get("user_models",{}).get(user.id)), None)
        await safe_edit_text(q, "✅ Обновлено.", group_model_inline(gkey, ukey, admin_rights))
        return
    if data == "g:trg":
        cfg = get_chat_cfg(context.bot_data, chat.id)
        trg = cfg.get("triggers", {})
        await safe_edit_text(
            q,
            "Когда бот должен отвечать (для всей группы):",
            group_triggers_inline(trg.get("mode", "mention"), trg.get("mute", False)),
        )
        return
    if data.startswith("g:T:"):
        cfg = get_chat_cfg(context.bot_data, chat.id)
        admin_rights = await is_group_admin(context, chat.id, user.id)
        if not admin_rights:
            await q.answer("Только админам.", show_alert=True)
            return
        _,_,part = data.split(":")
        if part in ("all","men","alw"):
            cfg["triggers"]["mode"] = {"all":"all","men":"mention","alw":"allow"}[part]
        elif part=="mt":
            cfg["triggers"]["mute"] = not cfg["triggers"].get("mute",False)
        elif part=="rst":
            cfg["triggers"] = {"mode":"mention","keywords":[],"allow":[],"mute":False}
        elif part=="kw":
            kws = cfg["triggers"].get("keywords", [])
            await safe_edit_text(
                q,
                "Ключевые слова:\n" + (", ".join(kws) if kws else "— пусто —"),
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("➕ Добавить", callback_data="g:Kw:+")],
                        [InlineKeyboardButton("🗑 Удалить", callback_data="g:Kw:-")],
                        [InlineKeyboardButton("⬅️ Назад", callback_data="g:trg")],
                    ]
                ),
            )
            return
        elif part=="aw":
            allow = cfg["triggers"].get("allow", [])
            await safe_edit_text(
                q,
                "Разрешённые:\n" + ("\n".join(map(str, allow)) if allow else "— пусто —"),
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("➕ Добавить ID", callback_data="g:Aw:+")],
                        [InlineKeyboardButton("🗑 Удалить ID", callback_data="g:Aw:-")],
                        [InlineKeyboardButton("🎯 Захват по reply (add)", callback_data="g:Aw:c+")],
                        [InlineKeyboardButton("🎯 Захват по reply (del)", callback_data="g:Aw:c-")],
                        [InlineKeyboardButton("⬅️ Назад", callback_data="g:trg")],
                    ]
                ),
            )
            return
        trg = cfg.get("triggers", {})
        await safe_edit_text(q, "Переключено.", group_triggers_inline(trg.get("mode","mention"), trg.get("mute",False)))
        return
    if data.startswith("g:Kw:"):
        op = data.split(":")[-1]
        context.user_data["await_kw_group"] = {"chat_id": chat.id, "op": ("add" if op=="+" else "del")}
        await safe_edit_text(q, "Введи слово/фразу.", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="g:trg")]]))
        return
    if data == "g:art":
        await safe_edit_text(q, "Выбери стиль генерации:", draw_style_inline("g", context.user_data.get("last_draw_style")))
        return
    if data.startswith("g:Aw:"):
        op = data.split(":")[-1]
        if op in ("+","-"):
            context.user_data["await_allow_group"] = {"chat_id": chat.id, "op": ("add" if op=="+" else "del")}
            await safe_edit_text(q, "Введи numeric user_id.", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="g:trg")]]))
            return
        else:
            context.user_data["capture_allow_group"] = {"chat_id": chat.id, "op": ("add" if op=="c+" else "del")}
            await safe_edit_text(q, "Ответь (Reply) на сообщение нужного пользователя в этом чате.", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="g:trg")]]))
            return
    # Admin (DM)
    async def send_admin_api_status() -> None:
        snapshot = await AI_KEY_MANAGER.snapshot()
        if not snapshot:
            text = "🔑 Состояние API-ключей:\n\nНет активных ключей."
        else:
            lines = ["🔑 Состояние API-ключей:\n"]
            for idx, entry in enumerate(snapshot, 1):
                star = "⭐ " if entry["preferred"] else ""
                rate = f"{entry['error_rate'] * 100:.0f}%"
                lines.append(f"{idx}. {star}{entry['display']}")
                lines.append(f"   Запросов: {entry['total']} | Успешно: {entry['success']} | Ошибок: {entry['errors']} ({rate})")
                last_status = entry["last_status"] if entry["last_status"] is not None else "—"
                if entry["last_error"]:
                    lines.append(f"   Последний статус: {last_status} — {entry['last_error']}")
                else:
                    lines.append(f"   Последний статус: {last_status}")
                if entry["last_updated"]:
                    lines.append(f"   Обновлено: {entry['last_updated']}")
                lines.append("")
            text = "\n".join(lines).strip()
        await safe_edit_text(q, text, admin_api_inline())

    async def send_admin_codes_overview() -> None:
        total = len(REDEMPTION_CODES)
        active = sum(1 for e in REDEMPTION_CODES.values() if e.get("status") == "active")
        used = sum(1 for e in REDEMPTION_CODES.values() if e.get("status") == "used")
        revoked = sum(1 for e in REDEMPTION_CODES.values() if e.get("status") == "revoked")
        text = (
            "🎟 *Коды персональных режимов*\n\n"
            f"• Всего сгенерировано: {total}\n"
            f"• Активные: {active}\n"
            f"• Использованные: {used}\n"
            f"• Отозванные: {revoked}\n\n"
            "Используй кнопки ниже, чтобы управлять кодами."
        )
        await safe_edit_text(q, text, admin_codes_inline())

    async def send_admin_codes_list(status: str, page: int = 0) -> None:
        status_map = {"active": "Активные", "used": "Использованные", "revoked": "Отозванные"}
        items = [entry for entry in REDEMPTION_CODES.values() if entry.get("status") == status]
        if status == "used":
            items.sort(key=lambda e: e.get("used_at") or e.get("created_at") or "", reverse=True)
        elif status == "revoked":
            items.sort(key=lambda e: e.get("revoked_at") or e.get("created_at") or "", reverse=True)
        else:
            items.sort(key=lambda e: e.get("created_at") or "", reverse=True)
        title = status_map.get(status, status)
        await safe_edit_text(
            q,
            f"🎟 {title} коды ({len(items)})",
            admin_codes_list_inline(items, status, page)
        )

    async def send_admin_code_detail(code: str, status: str, page: int) -> None:
        entry = REDEMPTION_CODES.get(code)
        if not entry:
            await q.answer("Код не найден.", show_alert=True)
            await send_admin_codes_overview()
            return
        text_lines = [
            "🎟 *Информация о коде*",
            f"• Код: `{entry.get('code')}`",
            f"• Статус: {entry.get('status')}",
            f"• Создан: {entry.get('created_at') or '—'}",
        ]
        if entry.get("comment"):
            text_lines.append(f"• Комментарий: {entry['comment']}")
        if entry.get("used_by"):
            text_lines.append(f"• Использован пользователем: {entry['used_by']} в {entry.get('used_at')}")
        if entry.get("revoked_by"):
            text_lines.append(f"• Отозван: {entry.get('revoked_at')} (админ {entry['revoked_by']})")
        await safe_edit_text(
            q,
            "\n".join(text_lines),
            admin_code_detail_inline(entry, entry.get("status", status), page)
        )

    if data == "a:home":
        await safe_edit_text(q, f"👑 Панель управления ({VERSION}):", admin_home_inline())
        return

    if data == "a:codes":
        await send_admin_codes_overview()
        return

    if data == "a:codes:new":
        context.user_data["await_admin_code_comment"] = {"chat_id": chat.id}
        await safe_edit_text(
            q,
            "Отправь комментарий для нового кода (или '-' для пустого).",
            admin_codes_inline()
        )
        return

    if data.startswith("a:codes:list:"):
        parts = data.split(":")
        status = parts[3] if len(parts) > 3 else "active"
        page = int(parts[4]) if len(parts) > 4 else 0
        await send_admin_codes_list(status, page)
        return

    if data.startswith("a:codes:show:"):
        parts = data.split(":")
        code = parts[3] if len(parts) > 3 else ""
        status = parts[4] if len(parts) > 4 else "active"
        page = int(parts[5]) if len(parts) > 5 else 0
        await send_admin_code_detail(code, status, page)
        return

    if data.startswith("a:codes:revoke:"):
        parts = data.split(":")
        code = parts[3] if len(parts) > 3 else ""
        page = int(parts[4]) if len(parts) > 4 else 0
        ok, msg_txt = await revoke_code(code, user.id)
        await q.answer(msg_txt, show_alert=not ok)
        entry = REDEMPTION_CODES.get(code)
        status = entry.get("status") if entry else "active"
        await send_admin_code_detail(code, status, page)
        return

    if data == "a:api":
        await send_admin_api_status()
        return

    if data == "a:api:refresh":
        await send_admin_api_status()
        return

    if data == "a:ch":
        await safe_edit_text(q, "Выбери категорию чатов:", admin_chats_inline())
        return

    if data.startswith("a:ch:"):
        parts = data.split(":")
        if len(parts) >= 4:
            cat = parts[2]
            page = int(parts[3])

            chats_info = context.bot_data.get("chats_info", {})
            items = []
            for chat_id, info in chats_info.items():
                if cat == "p" and info["type"] == "private":
                    items.append((chat_id, info["label"], info.get("last", ""), info.get("count", 0)))
                elif cat == "g" and info["type"] in ("group", "supergroup"):
                    items.append((chat_id, info["label"], info.get("last", ""), info.get("count", 0)))

            start_idx = page * PAGE_SIZE
            end_idx = start_idx + PAGE_SIZE
            page_items = items[start_idx:end_idx]

            title = "Личные" if cat == "p" else "Группы"
            await safe_edit_text(
                q,
                f"📂 {title} чаты (стр. {page+1}):",
                admin_chats_list_inline(page_items, page, cat),
            )
        return

    if data.startswith("a:ct:"):
        parts = data.split(":")
        if len(parts) >= 4:
            cid = int(parts[2])
            cat = parts[3]
            context.user_data["cfg_target"] = {"cid": cid, "cat": cat}

            chat_info = context.bot_data.get("chats_info", {}).get(cid, {})
            count = chat_info.get("count", 0)
            last = chat_info.get("last", "никогда")
            label = chat_info.get("label", f"Чат {cid}")

            info_text = (
                f"💬 Чат: {label}\n"
                f"ID: {cid}\n"
                f"Сообщений: {count}\n"
                f"Последнее: {last}"
            )
            await safe_edit_text(q, info_text, admin_view_chat_inline(cid, cat))
        return

    if data.startswith("a:cfg:"):
        parts = data.split(":")
        if len(parts) >= 4:
            cid = int(parts[2])
            cat = parts[3]
            context.user_data["cfg_target"] = {"cid": cid, "cat": cat}
            chat_info = context.bot_data.get("chats_info", {}).get(cid, {})
            if chat_info.get("type") in ("group", "supergroup"):
                await safe_edit_text(q, "⚙️ Настройки группы:", admin_group_cfg_inline())
            else:
                await safe_edit_text(q, "Для личных чатов специальных настроек нет.", admin_view_chat_inline(cid, cat))
        return

    if data == "a:ex":
        await safe_edit_text(q, "Выход из админ‑панели.")
        return

    if data == "a:lg":
        try:
            with open("logs.txt", "r", encoding="utf-8") as f:
                tail = "".join(f.readlines()[-20:])
            await safe_edit_text(
                q,
                "🧾 Последние логи:\n\n" + (tail[-3900:] or "— пусто —"),
                admin_home_inline(),
            )
        except Exception as e:
            await safe_edit_text(q, f"Ошибка чтения логов: {e}", admin_home_inline())
        return

    if data == "a:st":
        await send_admin_stats()
        return

    if data == "a:st:refresh":
        await send_admin_stats()
        return

    if data == "a:st:reset_daily":
        stats["active_today"] = []
        stats["last_reset"] = datetime.now().strftime("%Y-%m-%d")
        save_stats(stats)
        await q.answer("✅ Дневная статистика сброшена", show_alert=True)
        await send_admin_stats()
        return

    if data == "a:u":
        page = 0
        context.user_data["users_page"] = page
        user_items = collect_user_items()
        await safe_edit_text(
            q,
            f"👤 Пользователи ({len(user_items)}):",
            admin_users_list_inline(user_items, page),
        )
        return

    if data.startswith("a:u:p:"):
        try:
            page = int(data.split(":")[-1])
        except ValueError:
            page = 0
        if page < 0:
            page = 0
        context.user_data["users_page"] = page
        user_items = collect_user_items()
        await safe_edit_text(
            q,
            f"👤 Пользователи ({len(user_items)}):",
            admin_users_list_inline(user_items, page),
        )
        return

    if data.startswith("a:uv:"):
        parts = data.split(":")
        if len(parts) < 3:
            return
        try:
            uid = int(parts[2])
        except ValueError:
            return
        try:
            page = int(parts[3])
        except (IndexError, ValueError):
            page = context.user_data.get("users_page", 0)
        if page < 0:
            page = 0
        context.user_data["users_page"] = page
        users_info = context.bot_data.get("users_info", {})
        info = users_info.get(uid)
        if not info:
            user_items = collect_user_items()
            await safe_edit_text(
                q,
                "❌ Данные о пользователе не найдены.",
                admin_users_list_inline(user_items, page),
            )
            return

        chats_info = context.bot_data.get("chats_info", {})
        chat_cfg = context.bot_data.get("chat_cfg", {})
        banned_in = [cid for cid, cfg in chat_cfg.items() if uid in cfg.get("banned_users", [])]
        banned_labels = [
            chats_info.get(cid, {}).get("label", f"Чат {cid}")
            for cid in banned_in
        ]

        chats_list = info.get("chats", []) or []
        chat_stats = info.get("chat_stats", {}) or {}
        usage = []
        for cid in chats_list:
            count = chat_stats.get(str(cid), 0)
            label = chats_info.get(cid, {}).get("label", f"Чат {cid}")
            usage.append((count, label))
        usage.sort(reverse=True)
        chat_lines = [f"• {label} — {count}" for count, label in usage[:10]]

        all_messages = context.bot_data.get("all_messages", {})
        user_messages = []
        for chat_id, messages in all_messages.items():
            for msg_dict in messages[-200:]:
                if msg_dict.get("user_id") == uid:
                    user_messages.append((
                        msg_dict.get("timestamp", ""),
                        chat_id,
                        msg_dict.get("text", "")
                    ))
        user_messages.sort(key=lambda x: x[0], reverse=True)
        recent = user_messages[:5]
        recent_lines = []
        for ts, chat_id_value, msg_text in recent:
            chat_name = chats_info.get(chat_id_value, {}).get("label", f"Чат {chat_id_value}")
            snippet = (msg_text or "").replace("\n", " ").strip()
            if len(snippet) > 80:
                snippet = snippet[:77] + "…"
            if not snippet:
                snippet = "(без текста)"
            recent_lines.append(f"• {ts or '—'} — {chat_name}\n  {snippet}")

        full_name_parts = [
            part for part in [info.get("first_name"), info.get("last_name")] if part
        ]
        full_name = " ".join(full_name_parts) if full_name_parts else "—"
        username_display = f"@{info.get('username')}" if info.get("username") else "—"
        language = info.get("language_code") or "—"
        last_activity = info.get("last_activity", "—")
        first_seen = info.get("first_seen", "—")
        last_chat_label = info.get("last_chat_label") or "—"
        last_text = info.get("last_text")
        if last_text:
            last_text_display = last_text.replace("\n", " ").strip()
            if len(last_text_display) > 200:
                last_text_display = last_text_display[:197] + "…"
        else:
            last_text_display = "—"
        status_tags = []
        if uid == ADMIN_ID:
            status_tags.append("Админ")
        if uid == SPONSOR_ID:
            status_tags.append("Спонсор")
        status_tags.append("Бот" if info.get("is_bot") else "Живой")
        status_line = ", ".join(status_tags)

        detail_lines = [
            f"👤 {format_user_name(info, uid)}",
            f"ID: {uid}",
            f"Статус: {status_line}",
            f"Username: {username_display}",
            f"Имя: {full_name}",
            f"Язык: {language}",
            f"Всего сообщений: {info.get('messages', 0)}",
            f"Первое сообщение: {first_seen}",
            f"Последняя активность: {last_activity}",
            f"Последний чат: {last_chat_label}",
            f"Количество чатов: {len(chats_list)}",
            f"Блокировки: {'Да' if banned_in else 'Нет'}" + (f" ({len(banned_in)})" if banned_in else ""),
            f"Последний текст: {last_text_display}",
        ]

        if chat_lines:
            detail_lines.append("")
            detail_lines.append("Активность по чатам:")
            detail_lines.extend(chat_lines)

        if banned_labels:
            detail_lines.append("")
            detail_lines.append("Блокировки в чатах:")
            for label in banned_labels[:5]:
                detail_lines.append(f"• {label}")
            if len(banned_labels) > 5:
                detail_lines.append(f"… и ещё {len(banned_labels) - 5}")

        if recent_lines:
            detail_lines.append("")
            detail_lines.append("Последние сообщения:")
            detail_lines.extend(recent_lines)

        detail_text = "\n".join(detail_lines)
        await safe_edit_text(q, detail_text, admin_user_detail_inline(uid, page))
        return

    if data.startswith("a:v:"):
        parts = data.split(":")
        if len(parts) == 5:
            _, _, cid, count, cat = parts
            cid = int(cid)
            count = int(count)

            all_msgs = context.bot_data.get("all_messages", {}).get(cid, [])
            recent_msgs = context.bot_data.get("recent", {}).get(cid, [])
            lines = []
            for msg in (all_msgs + recent_msgs)[-count:]:
                if isinstance(msg, dict) and "role" in msg:
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    lines.append(f"{role_icon} [{msg.get('timestamp','')}] {msg.get('text','')}")
                else:
                    lines.append(f"[{msg.get('t','')}] {msg.get('who','')}: {msg.get('text','')}")

            out = "🧾 Все сообщения (пользователь + бот):\n\n" + ("\n".join(lines[-count:]) if lines else "— пусто —")
            await safe_edit_text(q, out[-3900:], admin_view_chat_inline(cid, cat))
        return

    if data.startswith("a:expt:"):
        parts = data.split(":")
        if len(parts) >= 4:
            _, _, cid, cat = parts[:4]
            cid = int(cid)
            rec = context.bot_data.get("recent", {}).get(cid, [])
            if not rec:
                await safe_edit_text(q, "Нет данных для экспорта.", admin_view_chat_inline(cid, cat))
                return
            os.makedirs("exports", exist_ok=True)
            fname = f"exports/chat_{cid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(
                    f"Export chat {cid} • {len(rec)} msgs • {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
                )
                for r in rec:
                    f.write(f"[{r.get('t','')}] {r.get('who','')}: {r.get('text','')}\n")
            with open(fname, "rb") as doc:
                await context.bot.send_document(
                    chat_id=q.message.chat_id,
                    document=InputFile(doc, filename=os.path.basename(fname)),
                    caption=f"Экспорт чата {cid}",
                )
            await safe_edit_text(q, "📤 Экспорт отправлен.", admin_view_chat_inline(cid, cat))
        return

    if data == "a:cfgb":
        target = context.user_data.get("cfg_target")
        if target:
            await safe_edit_text(q, "⚙️ Настройки группы:", admin_group_cfg_inline())
        else:
            await safe_edit_text(q, f"👑 Панель управления ({VERSION}):", admin_home_inline())
        return

    if data == "a:back":
        target = context.user_data.get("cfg_target")
        if target:
            await safe_edit_text(q, f"Чат {target['cid']}. Что показать?", admin_view_chat_inline(target["cid"], target["cat"]))
        else:
            await safe_edit_text(q, f"👑 Панель управления ({VERSION}):", admin_home_inline())
        return

    if data == "a:gm":
        target = context.user_data.get("cfg_target")
        if not target:
            await safe_edit_text(q, "Сначала выбери чат.", admin_home_inline())
            return
        cfg = get_chat_cfg(context.bot_data, target["cid"])
        gshort = MODE_MAP.get(cfg.get("group_mode", "normal"), "n")
        await safe_edit_text(q, "🎭 Режим группы:", admin_group_mode_inline(gshort))
        return

    if data.startswith("a:gm:"):
        val = data.split(":")[-1]
        target = context.user_data.get("cfg_target")
        if not target:
            await safe_edit_text(q, "Сначала выбери чат.", admin_home_inline())
            return
        cfg = get_chat_cfg(context.bot_data, target["cid"])
        cfg["group_mode"] = MODE_MAP_REV.get(val, "normal")
        gshort = MODE_MAP.get(cfg.get("group_mode", "normal"), "n")
        await safe_edit_text(q, "✅ Обновлено.", admin_group_mode_inline(gshort))
        return

    if data == "a:gD":
        target = context.user_data.get("cfg_target")
        if not target:
            await safe_edit_text(q, "Сначала выбери чат.", admin_home_inline())
            return
        cfg = get_chat_cfg(context.bot_data, target["cid"])
        gkey = next((k for k, v in MODEL_MAP.items() if v == cfg.get("group_model")), "ds")
        await safe_edit_text(q, "🤖 Модель группы:", admin_group_model_inline(gkey))
        return

    if data.startswith("a:gD:"):
        key = data.split(":")[-1]
        if key not in MODEL_MAP:
            await safe_edit_text(q, "Нет такой модели.", admin_group_cfg_inline())
            return
        target = context.user_data.get("cfg_target")
        if not target:
            await safe_edit_text(q, "Сначала выбери чат.", admin_home_inline())
            return
        cfg = get_chat_cfg(context.bot_data, target["cid"])
        cfg["group_model"] = MODEL_MAP[key]
        await safe_edit_text(q, "✅ Обновлено.", admin_group_model_inline(key))
        return

    if data == "a:gT":
        target = context.user_data.get("cfg_target")
        if not target:
            await safe_edit_text(q, "Сначала выбери чат.", admin_home_inline())
            return
        cfg = get_chat_cfg(context.bot_data, target["cid"])
        trg = cfg.get("triggers", {})
        await safe_edit_text(
            q,
            "🕹 Когда отвечать:",
            admin_group_triggers_inline(trg.get("mode", "mention"), trg.get("mute", False)),
        )
        return

    if data.startswith("a:gT:"):
        part = data.split(":")[-1]
        target = context.user_data.get("cfg_target")
        if not target:
            await safe_edit_text(q, "Сначала выбери чат.", admin_home_inline())
            return
        cfg = get_chat_cfg(context.bot_data, target["cid"])
        if part in ("all", "men", "alw"):
            cfg["triggers"]["mode"] = {"all": "all", "men": "mention", "alw": "allow"}[part]
        elif part == "mt":
            cfg["triggers"]["mute"] = not cfg["triggers"].get("mute", False)
        elif part == "rst":
            cfg["triggers"] = {"mode": "mention", "keywords": [], "allow": [], "mute": False}
        elif part == "kw":
            kws = cfg["triggers"].get("keywords", [])
            await safe_edit_text(
                q,
                "Ключевые слова:\n" + (", ".join(kws) if kws else "— пусто —"),
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("➕ Добавить", callback_data="a:gTk:+")],
                        [InlineKeyboardButton("🗑 Удалить", callback_data="a:gTk:-")],
                        [InlineKeyboardButton("⬅️ Назад", callback_data="a:gT")],
                    ]
                ),
            )
            return
        elif part == "aw":
            allow = cfg["triggers"].get("allow", [])
            await safe_edit_text(
                q,
                "Allowlist:\n" + ("\n".join(map(str, allow)) if allow else "— пусто —"),
                InlineKeyboardMarkup(
                    [
                        [InlineKeyboardButton("➕ Добавить ID", callback_data="a:gTa:+")],
                        [InlineKeyboardButton("🗑 Удалить ID", callback_data="a:gTa:-")],
                        [InlineKeyboardButton("⬅️ Назад", callback_data="a:gT")],
                    ]
                ),
            )
            return
        trg = cfg.get("triggers", {})
        await safe_edit_text(
            q,
            "✅ Обновлено.",
            admin_group_triggers_inline(trg.get("mode", "mention"), trg.get("mute", False)),
        )
        return
# Crash log viewer
    if data == "a:cr":
        tail = read_crash_tail(50) or "— всё чисто, ошибок не было —"
        await safe_edit_text(
            q,
            f"⚠️ Последние ошибки/краши:\n\n<code>{tail[-3900:]}</code>",
            InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("🧹 Очистить crash.log", callback_data="a:cr:cl")],
                    [InlineKeyboardButton("⬅️ Назад", callback_data="a:home")],
                ]
            ),
        )
        return
    if data == "a:cr:cl":
        try:
            open(CRASH_LOG, "w").close()
            await safe_edit_text(q, "🧹 crash.log очищен.", admin_home_inline())
        except Exception as e:
            await safe_edit_text(q, f"Не удалось очистить: {e}", admin_home_inline())
        return

    # Reports
    if data == "a:rp":
        reps = context.bot_data.get("reports", [])
        await safe_edit_text(q, f"📬 Репорты ({len(reps)}):", admin_reports_inline(reps, 0))
        return
    if data.startswith("a:rp:p"):
        page = int(data.split("p")[-1])
        reps = context.bot_data.get("reports", [])
        await safe_edit_text(q, f"📬 Репорты ({len(reps)}), стр. {page+1}", admin_reports_inline(reps, page))
        return
    if data == "a:rp:cl":
        context.bot_data["reports"] = []
        await safe_edit_text(q, "📬 Репорты очищены.", admin_home_inline())
        return
    if data.startswith("a:rp:d:"):
        idx = int(data.split(":")[-1])
        reps = context.bot_data.get("reports", [])
        if 0 <= idx < len(reps):
            reps.pop(idx)
        await safe_edit_text(q, "🗑 Удалено.", admin_reports_inline(reps, 0))
        return
    if data.startswith("a:rp:r:"):
        idx = int(data.split(":")[-1])
        reps = context.bot_data.get("reports", [])
        if 0 <= idx < len(reps):
            context.user_data["admin_reply_to"]  = reps[idx]["user_id"]
            context.user_data["admin_reply_idx"] = idx
            await safe_edit_text(q, f"Напиши ответ — отправлю пользователю id {reps[idx]['user_id']}", admin_report_view_inline(idx))
            return
    if data.startswith("a:rp:"):
        idx = int(data.split(":")[-1])
        reps = context.bot_data.get("reports", [])
        if 0 <= idx < len(reps):
            rep = reps[idx]
            txt = (f"📬 Репорт #{idx}\n"
                   f"от: {rep.get('who')} (id {rep.get('user_id')})\n"
                   f"чат: {rep.get('chat_id')} [{rep.get('chat_type')}]\n"
                   f"время: {rep.get('ts')}\n\n{rep.get('text')}")
            await safe_edit_text(q, txt, admin_report_view_inline(idx))
            return
    # Ban management
    if data == "a:ban":
        await safe_edit_text(q, "🚫 Управление банами:", admin_ban_inline())
        return

    # Массовая генерация изображений
    if data == "a:batch_gen":
        current_model = context.user_data.get("batch_model", "gemini")
        await safe_edit_text(q, "🎨 **Массовая генерация изображений**\n\nВыберите действие:", admin_batch_gen_inline(current_model))
        return

    if data == "a:batch:input":
        context.user_data["await_batch_objects"] = {"chat_id": chat.id}
        current_model = context.user_data.get("batch_model", "gemini")
        model_label = BATCH_MODEL_LABELS.get(current_model, "Неизвестная модель")
        await safe_edit_text(
            q,
            f"📝 **Введите список объектов для генерации**\n\n"
            f"🤖 **Текущая модель:** {model_label}\n"
            f"🎨 **Глобальный стиль:** Северная магия + реализм\n\n"
            "Формат:\n"
            "НАЗВАНИЕ (соотношение)\n"
            "Описание изображения...\n\n"
            "Пример:\n"
            "IMAGE_PROFILE_SCREEN (16:9)\n"
            "A clean cinematic HUD card layout, circular avatar frame on the left, frosty background with subtle snow particles, status bars and icons styled as glowing ice and neon lines, northern magic interface, blue and cyan dominant, PNG, transparent background, aspect ratio 16:9, no text, no actual numbers\n\n"
            "Напишите «Отмена» для возврата.",
            admin_batch_gen_inline(current_model)
        )
        return

    if data == "a:batch:model":
        current_model = context.user_data.get("batch_model", "gemini")
        await safe_edit_text(q, "🤖 **Выберите модель для генерации:**", admin_batch_model_inline(current_model))
        return

    if data.startswith("a:batch:setmodel:"):
        model_key = data.split(":")[-1]
        if model_key in BATCH_IMAGE_MODELS:
            context.user_data["batch_model"] = model_key
            model_label = BATCH_MODEL_LABELS.get(model_key, "Неизвестная модель")
            await safe_edit_text(q, f"✅ Модель установлена: {model_label}", admin_batch_model_inline(model_key))
        else:
            await safe_edit_text(q, "❌ Неизвестная модель.", admin_batch_model_inline())
        return

    if data == "a:batch:cost":
        await safe_edit_text(q, admin_batch_cost_inline().reply_markup, admin_batch_cost_inline())
        return

    if data == "a:batch:style":
        await safe_edit_text(q, "🎨 **Настройки глобального стиля**\n\nГлобальный стиль применяется ко всем изображениям:", admin_batch_style_inline())
        return

    if data == "a:batch:edit_style":
        context.user_data["await_batch_style"] = {"chat_id": chat.id}
        current_style = context.user_data.get("batch_global_style", BATCH_GLOBAL_STYLE)
        await safe_edit_text(
            q,
            f"✏️ **Изменение глобального стиля**\n\n"
            f"**Текущий стиль:**\n{current_style}\n\n"
            f"Отправьте новый глобальный стиль или «Отмена» для возврата.",
            admin_batch_style_inline()
        )
        return

    if data == "a:batch:view_style":
        current_style = context.user_data.get("batch_global_style", BATCH_GLOBAL_STYLE)
        text = f"👀 **Текущий глобальный стиль:**\n\n```\n{current_style}\n```"
        await safe_edit_text(q, text, admin_batch_style_inline())
        return

    if data == "a:batch:reset_style":
        context.user_data["batch_global_style"] = BATCH_GLOBAL_STYLE
        await safe_edit_text(q, "✅ Глобальный стиль сброшен по умолчанию.", admin_batch_style_inline())
        return

    if data == "a:batch:stats":
        # Показываем расширенную статистику
        total_images = stats.get("image_generations", 0)
        current_model = context.user_data.get("batch_model", "gemini")
        model_label = BATCH_MODEL_LABELS.get(current_model, "Неизвестная модель")
        current_style = context.user_data.get("batch_global_style", BATCH_GLOBAL_STYLE)
        
        style_preview = current_style[:100] + "..." if len(current_style) > 100 else current_style
        
        text = f"""
📊 **Статистика генераций изображений**

🎨 Всего сгенерировано: {total_images}

🤖 **Текущая модель:** {model_label}
🎨 **Глобальный стиль:** {style_preview}

💡 **Совет:** Используйте функцию массовой генерации для создания серии изображений из списка объектов.
        """
        await safe_edit_text(q, text, admin_batch_gen_inline(current_model))
        return

    if data.startswith("a:ban:"):
        action = data.split(":")[-1]
        if action == "user":
            context.user_data["await_ban_user"] = True
            await safe_edit_text(q, "Введите ID пользователя для бана:",
                               InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")]]))
            return
        elif action == "users_list":
            # Собираем список всех пользователей
            users_data = []
            user_ids = set()

            # Из статистики
            for user_id in stats.get("users", []):
                if user_id not in user_ids:
                    user_ids.add(user_id)
                    users_data.append({"user_id": user_id, "first_name": f"User {user_id}"})

            # Из чатов
            for chat_id, info in context.bot_data.get("chats_info", {}).items():
                if info["type"] == "private":
                    # Для приватных чатов chat_id = user_id
                    if chat_id not in user_ids:
                        user_ids.add(chat_id)
                        users_data.append({"user_id": chat_id, "first_name": info["label"].replace("👤 ", "")})

            if not users_data:
                await safe_edit_text(q, "📭 Список пользователей пуст", admin_ban_inline())
                return

            await safe_edit_text(q, f"👤 Выберите пользователя для бана ({len(users_data)}):",
                               admin_ban_users_list_inline(users_data, 0))
            return

        elif action == "group":
            context.user_data["await_ban_group"] = True
            await safe_edit_text(q, "Введите ID группы для бана:",
                               InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")]]))
            return

        elif action == "groups_list":
            # Собираем список групп
            groups_data = []
            for chat_id, info in context.bot_data.get("chats_info", {}).items():
                if info["type"] in ("group", "supergroup"):
                    groups_data.append({
                        "chat_id": chat_id,
                        "title": info["label"].replace("👥 ", "")
                    })

            if not groups_data:
                await safe_edit_text(q, "🏠 Список групп пуст", admin_ban_inline())
                return

            await safe_edit_text(q, f"👥 Выберите группу для бана ({len(groups_data)}):",
                               admin_ban_groups_list_inline(groups_data, 0))
            return

        elif action == "leave_group":
            # Собираем список групп для выхода
            groups_data = []
            for chat_id, info in context.bot_data.get("chats_info", {}).items():
                if info["type"] in ("group", "supergroup"):
                    groups_data.append({
                        "chat_id": chat_id,
                        "title": info["label"].replace("👥 ", "")
                    })

            if not groups_data:
                await safe_edit_text(q, "🏠 Нет групп для выхода", admin_ban_inline())
                return

            await safe_edit_text(q, f"🚪 Выберите группу для выхода ({len(groups_data)}):",
                               admin_ban_groups_list_inline(groups_data, 0, "leave"))
            return

        elif action == "list":
            # Показываем список банов
            banned_users = []
            banned_groups = []

            for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
                if cfg.get("is_banned"):
                    banned_groups.append(chat_id)
                banned_users.extend(cfg.get("banned_users", []))

            text = "🚫 Список банов:\n\n"
            text += f"👤 Забаненные пользователи: {len(set(banned_users))}\n"
            text += f"👥 Забаненные группы: {len(banned_groups)}\n"
            text += f"🚪 Выходы из групп: {len(context.bot_data.get('left_groups', []))}"

            await safe_edit_text(q, text, admin_ban_inline())
            return
    # Unban management
    if data.startswith("a:unban:"):
        action = data.split(":")[-1]
        if action == "user":
            context.user_data["await_unban_user"] = True
            await safe_edit_text(q, "Введите ID пользователя для разбана:",
                               InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")]]))
            return

        elif action == "users_list":
            # Собираем список забаненных пользователей
            banned_users_data = []
            banned_user_ids = set()

            # Собираем всех забаненных пользователей из всех чатов
            for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
                for user_id in cfg.get("banned_users", []):
                    if user_id not in banned_user_ids:
                        banned_user_ids.add(user_id)
                        # Ищем информацию о пользователе
                        user_info = {"user_id": user_id, "first_name": f"User {user_id}"}
                        # Пробуем найти в chats_info
                        for cid, info in context.bot_data.get("chats_info", {}).items():
                            if info["type"] == "private" and cid == user_id:
                                user_info["first_name"] = info["label"].replace("👤 ", "")
                                break
                        banned_users_data.append(user_info)

            if not banned_users_data:
                await safe_edit_text(q, "🔓 Нет забаненных пользователей", admin_ban_inline())
                return

            await safe_edit_text(q, f"🔓 Выберите пользователя для разбана ({len(banned_users_data)}):",
                               admin_unban_users_list_inline(banned_users_data, 0))
            return

        elif action == "group":
            context.user_data["await_unban_group"] = True
            await safe_edit_text(q, "Введите ID группы для разбана:",
                               InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="a:ban")]]))
            return

        elif action == "groups_list":
            # Собираем список забаненных групп
            banned_groups_data = []
            for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
                if cfg.get("is_banned", False):
                    group_info = {
                        "chat_id": chat_id,
                        "title": context.bot_data.get("chats_info", {}).get(chat_id, {}).get("label", f"Группа {chat_id}")
                    }
                    banned_groups_data.append(group_info)

            if not banned_groups_data:
                await safe_edit_text(q, "🔓 Нет забаненных групп", admin_ban_inline())
                return

            await safe_edit_text(q, f"🔓 Выберите группу для разбана ({len(banned_groups_data)}):",
                               admin_unban_groups_list_inline(banned_groups_data, 0))
            return

    # Обработка разбана пользователя из списка
    if data.startswith("a:unbanu:"):
        user_id = int(data.split(":")[-1])
        # Разбан пользователя во всех чатах
        unban_count = 0
        for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
            if "banned_users" in cfg and user_id in cfg["banned_users"]:
                cfg["banned_users"].remove(user_id)
                unban_count += 1

        await safe_edit_text(q, f"✅ Пользователь {user_id} разблокирован в {unban_count} чатах", admin_ban_inline())
        return

    # Обработка разбана группы из списка
    if data.startswith("a:unbang:"):
        chat_id = int(data.split(":")[-1])
        cfg = get_chat_cfg(context.bot_data, chat_id)
        if cfg.get("is_banned", False):
            cfg["is_banned"] = False
            group_title = context.bot_data.get("chats_info", {}).get(chat_id, {}).get("label", f"Группа {chat_id}")
            await safe_edit_text(q, f"✅ Группа {group_title} разблокирована", admin_ban_inline())
        else:
            await safe_edit_text(q, "❌ Эта группа не забанена", admin_ban_inline())
        return

    # Пагинация списка забаненных пользователей
    if data.startswith("a:unbanul:"):
        page = int(data.split(":")[-1])
        # Собираем список забаненных пользователей
        banned_users_data = []
        banned_user_ids = set()

        for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
            for user_id in cfg.get("banned_users", []):
                if user_id not in banned_user_ids:
                    banned_user_ids.add(user_id)
                    user_info = {"user_id": user_id, "first_name": f"User {user_id}"}
                    for cid, info in context.bot_data.get("chats_info", {}).items():
                        if info["type"] == "private" and cid == user_id:
                            user_info["first_name"] = info["label"].replace("👤 ", "")
                            break
                    banned_users_data.append(user_info)

        await safe_edit_text(q, f"🔓 Выберите пользователя для разбана ({len(banned_users_data)}):",
                           admin_unban_users_list_inline(banned_users_data, page))
        return

    # Пагинация списка забаненных групп
    if data.startswith("a:unbangl:"):
        page = int(data.split(":")[-1])
        # Собираем список забаненных групп
        banned_groups_data = []
        for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
            if cfg.get("is_banned", False):
                group_info = {
                    "chat_id": chat_id,
                    "title": context.bot_data.get("chats_info", {}).get(chat_id, {}).get("label", f"Группа {chat_id}")
                }
                banned_groups_data.append(group_info)

        await safe_edit_text(q, f"🔓 Выберите группу для разбана ({len(banned_groups_data)}):",
                           admin_unban_groups_list_inline(banned_groups_data, page))
        return

    # Обработка выбора пользователя для бана
    if data.startswith("a:banu:"):
        user_id = int(data.split(":")[-1])
        # Бан пользователя во всех чатах
        for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
            if "banned_users" not in cfg:
                cfg["banned_users"] = []
            if user_id not in cfg["banned_users"]:
                cfg["banned_users"].append(user_id)

        await safe_edit_text(q, f"✅ Пользователь {user_id} заблокирован во всех чатах", admin_ban_inline())
        return

    # Обработка выбора группы для бана
    if data.startswith("a:bang:"):
        chat_id = int(data.split(":")[-1])
        cfg = get_chat_cfg(context.bot_data, chat_id)
        cfg["is_banned"] = True

        # Выход из группы
        try:
            await context.bot.leave_chat(chat_id)
            # Сохраняем информацию о выходе
            if "left_groups" not in context.bot_data:
                context.bot_data["left_groups"] = []
            context.bot_data["left_groups"].append({
                "chat_id": chat_id,
                "banned_at": datetime.now().isoformat()
            })
            await safe_edit_text(q, f"✅ Группа {chat_id} заблокирована, бот вышел из группы", admin_ban_inline())
        except Exception as e:
            await safe_edit_text(q, f"❌ Ошибка при выходе из группы: {e}", admin_ban_inline())
        return

    # Обработка выхода из группы (без бана)
    if data.startswith("a:banlg:"):
        chat_id = int(data.split(":")[-1])

        # Выход из группы без бана
        try:
            group_title = context.bot_data.get("chats_info", {}).get(chat_id, {}).get("label", f"Группа {chat_id}")
            await context.bot.leave_chat(chat_id)
            # Сохраняем информацию о выходе
            if "left_groups" not in context.bot_data:
                context.bot_data["left_groups"] = []
            context.bot_data["left_groups"].append({
                "chat_id": chat_id,
                "title": group_title,
                "left_at": datetime.now().isoformat(),
                "banned": False
            })
            await safe_edit_text(q, f"✅ Бот вышел из группы {group_title}", admin_ban_inline())
        except Exception as e:
            await safe_edit_text(q, f"❌ Ошибка при выходе из группы: {e}", admin_ban_inline())
        return

    # Пагинация списка пользователей
    if data.startswith("a:banul:"):
        page = int(data.split(":")[-1])
        # Собираем список всех пользователей (аналогично выше)
        users_data = []
        user_ids = set()

        for user_id in stats.get("users", []):
            if user_id not in user_ids:
                user_ids.add(user_id)
                users_data.append({"user_id": user_id, "first_name": f"User {user_id}"})

        for chat_id, info in context.bot_data.get("chats_info", {}).items():
            if info["type"] == "private":
                if chat_id not in user_ids:
                    user_ids.add(chat_id)
                    users_data.append({"user_id": chat_id, "first_name": info["label"].replace("👤 ", "")})

        await safe_edit_text(q, f"👤 Выберите пользователя для бана ({len(users_data)}):",
                           admin_ban_users_list_inline(users_data, page))
        return

    # Пагинация списка групп
    if data.startswith("a:bangl:"):
        parts = data.split(":")
        page = int(parts[2])
        action = parts[3] if len(parts) > 3 else "ban"

        groups_data = []
        for chat_id, info in context.bot_data.get("chats_info", {}).items():
            if info["type"] in ("group", "supergroup"):
                groups_data.append({
                    "chat_id": chat_id,
                    "title": info["label"].replace("👥 ", "")
                })

        title = "бана" if action == "ban" else "выхода"
        await safe_edit_text(q, f"👥 Выберите группу для {title} ({len(groups_data)}):",
                           admin_ban_groups_list_inline(groups_data, page, action))
        return

# ---------- Batch Image Generation Functions ----------
def parse_object_list(text: str) -> List[Dict[str, Any]]:
    """Парсит текст с объектами для генерации изображений"""
    objects = []
    lines = text.strip().split('\n')
    current_object = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Проверяем, является ли строка названием объекта (содержит (соотношение))
        if '(' in line and ')' in line and line.upper().startswith('IMAGE_'):
            if current_object:
                objects.append(current_object)
            # Парсим название и соотношение
            parts = line.split('(')
            name = parts[0].strip()
            ratio_part = parts[1].split(')')[0].strip()
            current_object = {
                "name": name,
                "ratio": ratio_part,
                "description": ""
            }
        elif current_object:
            # Это описание для текущего объекта
            if current_object["description"]:
                current_object["description"] += "\n"
            current_object["description"] += line

    if current_object:
        objects.append(current_object)

    return objects

async def bg_batch_generate_images(context: ContextTypes.DEFAULT_TYPE, chat_id: int, waiting_msg_id: int, objects: List[Dict[str, Any]], style: str, model_key: str):
    """Фоновая генерация изображений для списка объектов"""
    try:
        total_objects = len(objects)
        global_style = context.user_data.get("batch_global_style", BATCH_GLOBAL_STYLE)

        # Обновляем сообщение о прогрессе
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=waiting_msg_id,
            text=f"🎨 Начинаю генерацию {total_objects} изображений...\n\nМодель: {BATCH_MODEL_LABELS.get(model_key, model_key)}\nСтиль: {style}"
        )

        generated_images = []
        failed_objects = []
        start_time = datetime.now()

        for i, obj in enumerate(objects, 1):
            try:
                # Формируем промпт
                prompt_parts = []
                if obj.get("description"):
                    prompt_parts.append(obj["description"])
                
                # Добавляем глобальный стиль
                if global_style:
                    prompt_parts.append(global_style)
                
                # Добавляем стиль если указан
                if style and style != "normal" and style != "global":
                    style_descriptions = {
                        "flirt": "romantic, elegant, soft lighting, cinematic",
                        "land": "landscape, scenic, nature, dramatic sky, wide angle",
                        "fantasy": "fantasy art, magical, mythical, dramatic lighting, detailed"
                    }
                    if style in style_descriptions:
                        prompt_parts.append(style_descriptions[style])

                full_prompt = ", ".join(prompt_parts)
                aspect_ratio = obj.get("ratio", "16:9")

                # Генерируем изображение через существующую функцию
                image_data = await generate_image_openrouter(
                    prompt=full_prompt,
                    style="normal",  # Стиль уже в промпте
                    aspect_ratio=aspect_ratio,
                    model=BATCH_IMAGE_MODELS.get(model_key)
                )

                if image_data:
                    generated_images.append({
                        "name": obj['name'],
                        "data": image_data,
                        "ratio": aspect_ratio
                    })
                    
                    # Увеличиваем счётчик генераций
                    stats["image_generations"] = stats.get("image_generations", 0) + 1
                    save_stats(stats)

                    # Обновляем прогресс
                    elapsed = (datetime.now() - start_time).seconds
                    eta = (elapsed / i) * (total_objects - i) if i > 0 else 0
                    progress_text = (
                        f"🎨 Генерация: {i}/{total_objects}\n"
                        f"✅ {obj['name']} - готово\n"
                        f"⏱ Прошло: {elapsed}с | Осталось: ~{int(eta)}с"
                    )
                    if failed_objects:
                        progress_text += f"\n❌ Ошибок: {len(failed_objects)}"
                    
                    try:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=waiting_msg_id,
                            text=progress_text
                        )
                    except BadRequest:
                        pass  # Сообщение не изменилось
                else:
                    failed_objects.append(obj["name"])
                    logger.warning(f"Failed to generate image for {obj['name']}")

            except Exception as e:
                failed_objects.append(obj["name"])
                logger.error(f"Error generating image for {obj['name']}: {e}")
                continue
            
            # Небольшая пауза между запросами чтобы не перегрузить API
            await asyncio.sleep(1)

        # Отправляем результаты
        total_time = (datetime.now() - start_time).seconds
        
        if generated_images:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=waiting_msg_id,
                text=f"✅ Генерация завершена!\n\n🎨 Сгенерировано: {len(generated_images)}\n❌ Ошибок: {len(failed_objects)}\n⏱ Время: {total_time}с\n\n📤 Отправляю изображения..."
            )

            for img in generated_images:
                try:
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=img["data"],
                        caption=f"🖼 {img['name']}\n📐 {img['ratio']}"
                    )
                    await asyncio.sleep(0.5)  # Пауза между отправками
                except Exception as e:
                    logger.error(f"Failed to send image {img['name']}: {e}")
            
            # Итоговое сообщение
            summary = (
                f"🎯 **Массовая генерация завершена!**\n\n"
                f"📊 Результаты:\n"
                f"• Всего объектов: {total_objects}\n"
                f"• Успешно: ✅ {len(generated_images)}\n"
                f"• Ошибок: ❌ {len(failed_objects)}\n"
                f"• Время: {total_time}с\n\n"
                f"⚡ Эффективность: {(len(generated_images)/total_objects*100):.1f}%"
            )
            if failed_objects:
                summary += f"\n\n❌ Не удалось: {', '.join(failed_objects[:5])}"
                if len(failed_objects) > 5:
                    summary += f" и ещё {len(failed_objects) - 5}"
            
            await context.bot.send_message(chat_id=chat_id, text=summary)
        else:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=waiting_msg_id,
                text=f"❌ Не удалось сгенерировать ни одного изображения.\n\nПроверьте баланс API или попробуйте другую модель."
            )

    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=waiting_msg_id,
                text=f"❌ Критическая ошибка: {str(e)[:200]}"
            )
        except:
            pass

# ---------- Messages ----------
async def on_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    msg, chat, user = update.message, update.message.chat, update.message.from_user
    text = msg.text.strip()

    ensure_store(context.bot_data, chat.id, chat.type, chat_label(chat, user))

    # ПРОВЕРКА БАНА
    cfg = get_chat_cfg(context.bot_data, chat.id)
    if user.id in cfg.get("banned_users", []):
        await msg.reply_text("🚫 Вы заблокированы в этом боте.")
        return

    if cfg.get("is_banned", False) and chat.type != "private":
        await msg.reply_text("🚫 Бот заблокирован в этой группе.")
        return

    # ---- Спец. состояния (админ и персональные режимы) ----
    if chat.type == "private":
        admin_code_state = context.user_data.get("await_admin_code_comment")
        if admin_code_state and admin_code_state.get("chat_id") == chat.id and user.id == ADMIN_ID:
            context.user_data.pop("await_admin_code_comment", None)
            if text.lower() in ("отмена", "cancel"):
                await msg.reply_text("Создание кода отменено.")
                return
            comment = None if text.strip() == "-" else text.strip()
            entry = await create_redeem_code(user.id, comment)
            await msg.reply_text(
                "✅ Код создан.\n"
                f"```\n{entry['code']}\n```\n"
                f"Комментарий: {entry.get('comment') or '—'}",
                parse_mode="Markdown",
                reply_markup=admin_codes_inline()
            )
            return

        code_state = context.user_data.get("await_custom_code")
        if code_state and code_state.get("chat_id") == chat.id:
            context.user_data.pop("await_custom_code", None)
            if text.lower() in ("отмена", "cancel"):
                await msg.reply_text("Ввод кода отменён.", reply_markup=user_reply_menu())
                return
            success, message = await redeem_code_for_user(user.id, text)
            await msg.reply_text(message, reply_markup=user_reply_menu())
            if success and ADMIN_ID:
                try:
                    await context.bot.send_message(
                        chat_id=ADMIN_ID,
                        text=f"🎟 Пользователь {user.id} активировал код персонального режима."
                    )
                except Exception as e:
                    logger.warning(f"notify admin about code redeem failed: {e}")
            return

        builder = context.user_data.get("custom_builder")
        if builder and builder.get("chat_id") == chat.id:
            if text.lower() in ("отмена", "cancel"):
                context.user_data.pop("custom_builder", None)
                await msg.reply_text("Создание персонального режима отменено.", reply_markup=user_reply_menu())
                return

            step = builder.get("step")
            if step == "title":
                title = text.strip()
                if not (1 <= len(title) <= 48):
                    await msg.reply_text(
                        "Название должно быть от 1 до 48 символов. Попробуй снова или напиши «Отмена».",
                        reply_markup=user_reply_menu()
                    )
                    return
                context.user_data["custom_builder"]["title"] = title
                context.user_data["custom_builder"]["step"] = "prompt"
                await msg.reply_text(
                    "Теперь отправь текст системного промпта (минимум 10 символов).\n"
                    "Напиши «Отмена», чтобы отменить.",
                    reply_markup=user_reply_menu()
                )
                return
            if step == "prompt":
                title = builder.get("title", "Персональный режим")
                success, message = await update_custom_prompt(user.id, title, text)
                if success:
                    context.user_data.pop("custom_builder", None)
                    await msg.reply_text(message, reply_markup=user_reply_menu())
                else:
                    context.user_data["custom_builder"]["step"] = "prompt"
                    await msg.reply_text(f"{message}\nПопробуй снова или напиши «Отмена».", reply_markup=user_reply_menu())
                return

    # ---- Обработка генерации изображений, если ждём промпт ----
    dr = context.user_data.get("draw")
    if dr:
        if dr.get("chat") == chat.id:
            style = dr.get("style","normal")
            context.user_data.pop("draw", None)
            
            # Rate-limiting для изображений (админ и спонсор без лимита)
            if user.id != ADMIN_ID and user.id != SPONSOR_ID:
                allowed, error_msg = check_rate_limit(user.id, "image")
                if not allowed:
                    await msg.reply_text(error_msg, reply_markup=user_reply_menu() if chat.type == "private" else None)
                    return
            
            waiting = await msg.reply_text("🎨 Генерирую изображение... ⌛")
            add_pending(chat.id, user.id, f"[DRAW] {text}", waiting.message_id)
            asyncio.create_task(bg_generate_image(context, chat.id, waiting.message_id, text, style, user_id=user.id))
            return

    # ---- Статистика/история ----
    # Инициализируем отсутствующие ключи
    for key in ["private_chats", "group_chats", "user_messages", "bot_messages", "image_generations", "active_today"]:
        if key not in stats:
            stats[key] = [] if key.endswith("s") else 0

    if "last_reset" not in stats:
        stats["last_reset"] = datetime.now().strftime("%Y-%m-%d")

    if user.id not in stats["users"]:
        stats["users"].append(user.id)
        # Уведомляем админа о новом пользователе
        asyncio.create_task(notify_new_user(context.bot, user, len(stats["users"])))

    if chat.id not in stats["chats"]:
        stats["chats"].append(chat.id)

    # Обновляем списки типов чатов
    if chat.type == "private":
        if chat.id not in stats["private_chats"]:
            stats["private_chats"].append(chat.id)
    elif chat.type in ("group", "supergroup"):
        if chat.id not in stats["group_chats"]:
            stats["group_chats"].append(chat.id)

    stats["messages"] += 1

    # Обновляем активных сегодня
    today = datetime.now().strftime("%Y-%m-%d")
    if stats.get("last_reset") != today:
        stats["active_today"] = []
        stats["last_reset"] = today

    if chat.type == "private" and user.id not in stats["active_today"]:
        stats["active_today"].append(user.id)

    save_stats(stats)

    # Обновляем информацию о пользователях для админ-панели
    users_info = context.bot_data.setdefault("users_info", {})
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    uinfo = users_info.setdefault(user.id, {})
    if "first_seen" not in uinfo:
        uinfo["first_seen"] = now_str
    if user.first_name:
        uinfo["first_name"] = user.first_name
    else:
        uinfo.setdefault("first_name", "")
    if user.last_name:
        uinfo["last_name"] = user.last_name
    else:
        uinfo.setdefault("last_name", "")
    if user.username:
        uinfo["username"] = user.username
    else:
        uinfo.setdefault("username", None)
    if getattr(user, "language_code", None):
        uinfo["language_code"] = user.language_code
    uinfo["is_bot"] = getattr(user, "is_bot", False)
    uinfo["messages"] = uinfo.get("messages", 0) + 1
    chats_list = uinfo.setdefault("chats", [])
    if chat.id not in chats_list:
        chats_list.append(chat.id)
    chat_stats = uinfo.setdefault("chat_stats", {})
    chat_key = str(chat.id)
    chat_stats[chat_key] = chat_stats.get(chat_key, 0) + 1
    chats_info = context.bot_data.get("chats_info", {})
    chat_label_value = chats_info.get(chat.id, {}).get("label") or chat_label(chat, user)
    uinfo["last_activity"] = now_str
    uinfo["last_chat_id"] = chat.id
    uinfo["last_chat_label"] = chat_label_value
    uinfo["last_chat_type"] = chat.type
    if text:
        uinfo["last_text"] = text[:400]
    uinfo["last_message_id"] = msg.message_id

    # СОХРАНЯЕМ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ
    add_to_memory(context, "user", text, chat.id, user.id)

    context.bot_data["chats_info"][chat.id]["last"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    context.bot_data["chats_info"][chat.id]["count"] += 1
    rl = context.bot_data["recent"][chat.id]
    rl.append({"t": datetime.now().strftime("%H:%M"), "who": (user.username or user.first_name or "user"), "text": text})
    if len(rl) > RECENT_LIMIT: context.bot_data["recent"][chat.id] = rl[-RECENT_LIMIT:]

    # Кнопки ЛС
    if text == "📋 Меню" and chat.type=="private":
        await msg.reply_text("Главное меню:", reply_markup=user_home_inline(context.bot.username, user.id))
        return
    if text == "🎨 Генерация изображений" and chat.type=="private":
        await msg.reply_text("Выбери стиль генерации:", reply_markup=draw_style_inline("u", context.user_data.get("last_draw_style")))
        return
    if text == "🧹 Обнулить диалог" and chat.type=="private":
        context.user_data["memory"] = []
        await msg.reply_text("🧹 Память очищена.", reply_markup=user_reply_menu())
        return
    if text == "ℹ️ Инфо" and chat.type=="private":
        info_text = (
            f"🤖 *DeepSeek AI Assistant* {VERSION}\n\n"
            f"✨ *Возможности:*\n"
            f"• 💬 Умный диалог с ИИ\n"
            f"• 🎨 Генерация изображений (/draw)\n"
            f"• 🎭 4 режима общения (нормальный, грубый, берсерк, пошлый)\n"
            f"• 🤖 Выбор моделей (DeepSeek, Mistral, OpenChat, Llama)\n"
            f"• 👥 Работа в группах и личных чатах\n"
            f"• 🔧 Гибкие настройки для каждого пользователя\n\n"

            f"⚙️ *Техническая информация:*\n"
            f"• Модели: DeepSeek, Mistral, OpenAI, Anthropic и другие\n"
            f"• Поддержка изображений: Gemini Vision\n\n"

            f"🆘 *Поддержка:*\n"
            f"• Нажми «📩 Поддержка» для связи с админом\n"
            f"• Сообщи о багах и предложи улучшения\n"

            f"📞 *Контакты:* @pirlesa (создатель)"
        )
        await msg.reply_text(info_text, parse_mode="Markdown", reply_markup=user_reply_menu())
        return

    # Group triggers для текста: если не хотим отвечать — выходим
    if chat.type in ("group","supergroup") and text.startswith("/draw"):
        pass
    elif chat.type in ("group","supergroup") and not should_respond_in_group(msg, context):
        return

    # Admin reply to user (DM or fallback to chat)
    if update.effective_user.id == ADMIN_ID and context.user_data.get("admin_reply_to"):
        to_user = context.user_data.pop("admin_reply_to")
        idx = context.user_data.pop("admin_reply_idx", None)
        try:
            await context.bot.send_message(chat_id=to_user, text=f"✉️ Ответ от админа:\n\n{text}")
            await msg.reply_text("✅ Отправлено в личные сообщения.", reply_markup=(user_reply_menu() if chat.type=="private" else None))
        except Forbidden:
            rep = None
            if idx is not None:
                reps = context.bot_data.get("reports", [])
                if 0 <= idx < len(reps): rep = reps[idx]
            if rep:
                mention = f"[ответ для](tg://user?id={to_user})"
                payload = f"✉️ {mention}:\n\n{text}"
                try:
                    if rep.get("msg_id"):
                        await context.bot.send_message(chat_id=rep["chat_id"], text=payload, parse_mode="Markdown", reply_to_message_id=rep["msg_id"])
                    else:
                        await context.bot.send_message(chat_id=rep["chat_id"], text=payload, parse_mode="Markdown")
                    await msg.reply_text("☑️ Отправлено в исходный чат (ЛС недоступны).")
                except Exception as e:
                    await msg.reply_text(f"⚠️ Не удалось отправить ни в ЛС, ни в чат: {e}")
            else:
                await msg.reply_text("⚠️ Нельзя написать в ЛС и нет данных исходного чата для репорта.")
        except Exception as e:
            await msg.reply_text(f"⚠️ Не удалось отправить: {e}", reply_markup=(user_reply_menu() if chat.type=="private" else None))
        return

    # Admin ban commands
    if update.effective_user.id == ADMIN_ID and context.user_data.get("await_ban_user"):
        context.user_data.pop("await_ban_user")
        try:
            user_id = int(text)
            # Бан пользователя во всех чатах
            for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
                if "banned_users" not in cfg:
                    cfg["banned_users"] = []
                if user_id not in cfg["banned_users"]:
                    cfg["banned_users"].append(user_id)

            await msg.reply_text(f"✅ Пользователь {user_id} заблокирован во всех чатах")
        except ValueError:
            await msg.reply_text("❌ Укажите числовой ID пользователя")
        return

    if update.effective_user.id == ADMIN_ID and context.user_data.get("await_ban_group"):
        context.user_data.pop("await_ban_group")
        try:
            group_id = int(text)
            cfg = get_chat_cfg(context.bot_data, group_id)
            cfg["is_banned"] = True

            # Выход из группы
            try:
                await context.bot.leave_chat(group_id)
                # Сохраняем информацию о выходе
                if "left_groups" not in context.bot_data:
                    context.bot_data["left_groups"] = []
                context.bot_data["left_groups"].append({
                    "chat_id": group_id,
                    "banned_at": datetime.now().isoformat()
                })
                await msg.reply_text(f"✅ Группа {group_id} заблокирована, бот вышел из группы")
            except Exception as e:
                await msg.reply_text(f"❌ Ошибка при выходе из группы: {e}")
        except ValueError:
            await msg.reply_text("❌ Укажите числовой ID группы")
        return

    # Admin unban commands
    if update.effective_user.id == ADMIN_ID and context.user_data.get("await_unban_user"):
        context.user_data.pop("await_unban_user")
        try:
            user_id = int(text)
            # Разбан пользователя во всех чатах
            unban_count = 0
            for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
                if "banned_users" in cfg and user_id in cfg["banned_users"]:
                    cfg["banned_users"].remove(user_id)
                    unban_count += 1

            if unban_count > 0:
                await msg.reply_text(f"✅ Пользователь {user_id} разблокирован в {unban_count} чатах")
            else:
                await msg.reply_text(f"ℹ️ Пользователь {user_id} не был забанен")
        except ValueError:
            await msg.reply_text("❌ Укажите числовой ID пользователя")
        return

    if update.effective_user.id == ADMIN_ID and context.user_data.get("await_unban_group"):
        context.user_data.pop("await_unban_group")
        try:
            group_id = int(text)
            cfg = get_chat_cfg(context.bot_data, group_id)
            if cfg.get("is_banned", False):
                cfg["is_banned"] = False
                await msg.reply_text(f"✅ Группа {group_id} разблокирована")
            else:
                await msg.reply_text(f"ℹ️ Группа {group_id} не была забанена")
        except ValueError:
            await msg.reply_text("❌ Укажите числовой ID группы")
        return

    # Admin inputs for keywords/allow
    if update.effective_user.id == ADMIN_ID and context.user_data.get("await_kw_group"):
        task = context.user_data.pop("await_kw_group")
        cfg = get_chat_cfg(context.bot_data, task["chat_id"])
        kw = text.strip()
        if kw:
            if task["op"]=="add":
                if kw not in cfg["triggers"]["keywords"]: cfg["triggers"]["keywords"].append(kw)
                await msg.reply_text(f"✅ Добавлено ключевое «{kw}».")
            else:
                if kw in cfg["triggers"]["keywords"]: cfg["triggers"]["keywords"].remove(kw)
                await msg.reply_text(f"🗑 Удалено ключевое «{kw}».")
        else:
            await msg.reply_text("⚠️ Пустое слово не принято.")
        return
    if update.effective_user.id == ADMIN_ID and context.user_data.get("await_allow_group"):
        task = context.user_data.pop("await_allow_group")
        cfg = get_chat_cfg(context.bot_data, task["chat_id"])
        try:
            uid = int(text)
            if task["op"]=="add":
                if uid not in cfg["triggers"]["allow"]: cfg["triggers"]["allow"].append(uid)
                await msg.reply_text(f"✅ Добавлен {uid} в allowlist.")
            else:
                if uid in cfg["triggers"]["allow"]: cfg["triggers"]["allow"].remove(uid)
                await msg.reply_text(f"🗑 Удалён {uid} из allowlist.")
        except ValueError:
            await msg.reply_text("⚠️ Нужен числовой user_id.")
        return
    if update.effective_user.id == ADMIN_ID and context.user_data.get("capture_allow_group"):
        task = context.user_data.get("capture_allow_group")
        if chat.id == task.get("chat_id"):
            if msg.reply_to_message and msg.reply_to_message.from_user:
                target_user = msg.reply_to_message.from_user.id
                cfg = get_chat_cfg(context.bot_data, chat.id)
                if task["op"]=="add":
                    if target_user not in cfg["triggers"]["allow"]:
                        cfg["triggers"]["allow"].append(target_user)
                    await msg.reply_text(f"✅ {target_user} добавлен в allowlist.")
                else:
                    if target_user in cfg["triggers"]["allow"]:
                        cfg["triggers"]["allow"].remove(target_user)
                    await msg.reply_text(f"🗑 {target_user} удалён из allowlist.")
                context.user_data.pop("capture_allow_group", None)
                return
            else:
                await msg.reply_text("Ответь (Reply) на сообщение нужного пользователя.")
                return

    # Массовая генерация изображений - обработка ввода списка
    if update.effective_user.id == ADMIN_ID and context.user_data.get("await_batch_objects"):
        batch_state = context.user_data.pop("await_batch_objects")
        if batch_state.get("chat_id") == chat.id:
            if text.lower() in ("отмена", "cancel"):
                await msg.reply_text("Массовая генерация отменена.", reply_markup=admin_batch_gen_inline())
                return

            # Парсим список объектов
            try:
                objects_list = parse_object_list(text)
                if not objects_list:
                    await msg.reply_text("❌ Не удалось распознать объекты. Проверьте формат.", reply_markup=admin_batch_gen_inline())
                    return

                # Просим выбрать стиль
                context.user_data["batch_generation"] = {
                    "objects": objects_list,
                    "chat_id": chat.id
                }

                # Меню выбора стиля с кнопкой быстрой генерации
                quick_gen_markup = InlineKeyboardMarkup([
                    [InlineKeyboardButton("⚡ Быстрая генерация (глобальный стиль)", callback_data="a:batch:quick_gen")],
                    [InlineKeyboardButton("🎨 Выбрать стиль вручную", callback_data="a:batch:select_style")]
                ])

                await msg.reply_text(
                    f"✅ Распознано {len(objects_list)} объектов.\n\n"
                    f"Выберите вариант генерации:",
                    reply_markup=quick_gen_markup
                )
                return

            except Exception as e:
                logger.error(f"Ошибка парсинга списка объектов: {e}")
                await msg.reply_text(f"❌ Ошибка при обработке списка: {str(e)[:100]}", reply_markup=admin_batch_gen_inline())
                return

    # Обработка изменения глобального стиля для массовой генерации
    if update.effective_user.id == ADMIN_ID and context.user_data.get("await_batch_style"):
        style_state = context.user_data.pop("await_batch_style")
        if style_state.get("chat_id") == chat.id:
            if text.lower() in ("отмена", "cancel"):
                await msg.reply_text("Изменение стиля отменено.", reply_markup=admin_batch_style_inline())
                return
            
            # Сохраняем новый глобальный стиль
            context.user_data["batch_global_style"] = text.strip()
            await msg.reply_text("✅ Глобальный стиль обновлен.", reply_markup=admin_batch_style_inline())
            return

    # Support report (DM or group)
    if context.user_data.get("support_mode"):
        context.user_data["support_mode"] = False
        report = {
            "user_id": user.id, "who": user.username or user.first_name or "user",
            "chat_id": chat.id, "chat_type": chat.type, "text": text,
            "msg_id": msg.message_id,
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        context.bot_data.setdefault("reports", []).append(report)
        if ADMIN_ID:
            try:
                await context.bot.send_message(chat_id=ADMIN_ID,
                    text=(f"📬 Репорт от {report['who']} (id {report['user_id']})\n"
                          f"чат {report['chat_id']} [{report['chat_type']}] • {report['ts']}\n\n{report['text']}"))
            except Exception: pass
        await msg.reply_text("✅ Репорт отправлен админу. Спасибо!", reply_markup=(user_reply_menu() if chat.type=="private" else None))
        return

    # Resolve mode/model
    is_private = chat.type=="private"
    mode = resolve_effective_mode(context, chat.id, user.id, is_private)
    model= resolve_effective_model(context, chat.id, user.id, is_private)
    if not mode:
        if is_private:
            await msg.reply_text("⚙️ Сначала выбери режим («📋 Меню → 🎭 Выбрать режим»).", reply_markup=user_reply_menu())
            return
        else:
            mode = get_chat_cfg(context.bot_data, chat.id).get("group_mode","normal")

    # ========== RATE LIMITING ==========
    # Админ и спонсор не имеют лимитов
    if user.id != ADMIN_ID and user.id != SPONSOR_ID:
        allowed, error_msg = check_rate_limit(user.id, "message")
        if not allowed:
            await msg.reply_text(error_msg, reply_markup=user_reply_menu() if is_private else None)
            return

    waiting = await msg.reply_text("⚙️ Генерирую ответ... ⌛")
    add_pending(chat.id, user.id, text, waiting.message_id)
    asyncio.create_task(bg_generate(context, chat.id, waiting.message_id, text, mode, model, user.id))

# ---------- Startup tasks ----------
async def notify_reboot(app):
    try:
        if ADMIN_ID:
            tail = read_crash_tail(1).strip()
            msg = f"✅ Бот запущен {VERSION} • {datetime.now():%Y-%m-%d %H:%M}"
            if tail:
                msg += f"\nПоследняя строка crash.log: {tail}"
            await app.bot.send_message(chat_id=ADMIN_ID, text=msg)
    except Exception as e:
        logger.warning(f"notify admin on boot failed: {e}")
    pend = load_pending()
    if pend:
        for item in pend[-20:]:
            try:
                await app.bot.send_message(
                    chat_id=item["chat"],
                    text=("⚠️ Бот был временно недоступен и перезапустился.\n"
                          "Если хочешь, повтори последний запрос:\n\n"
                          f"{item.get('text','(пусто)')}")
                )
            except Exception as e:
                logger.warning(f"notify user pending failed: chat {item.get('chat')}: {e}")
        save_pending_list([])

# ---------- Ban Commands ----------
async def ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для бана пользователя: /ban <user_id> или /ban в ответ на сообщение"""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Нет доступа.")
        return

    target_user_id = None

    # Если есть аргументы
    if context.args:
        try:
            target_user_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Укажите ID пользователя: /ban <user_id>")
            return

    # Если ответ на сообщение
    elif update.message.reply_to_message:
        target_user_id = update.message.reply_to_message.from_user.id

    if not target_user_id:
        await update.message.reply_text("❌ Укажите ID пользователя или ответьте на сообщение")
        return

    # Бан пользователя во всех чатах
    for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
        if "banned_users" not in cfg:
            cfg["banned_users"] = []
        if target_user_id not in cfg["banned_users"]:
            cfg["banned_users"].append(target_user_id)

    await update.message.reply_text(f"✅ Пользователь {target_user_id} заблокирован во всех чатах")

async def unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для разбана пользователя"""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Нет доступа.")
        return

    if not context.args:
        await update.message.reply_text("❌ Укажите ID пользователя: /unban <user_id>")
        return

    try:
        target_user_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Укажите числовой ID пользователя")
        return

    # Разбан пользователя во всех чатах
    for chat_id, cfg in context.bot_data.get("chat_cfg", {}).items():
        if "banned_users" in cfg and target_user_id in cfg["banned_users"]:
            cfg["banned_users"].remove(target_user_id)

    await update.message.reply_text(f"✅ Пользователь {target_user_id} разблокирован")

async def ban_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Бан группы и выход из неё"""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Нет доступа.")
        return

    chat = update.effective_chat
    if chat.type == "private":
        await update.message.reply_text("❌ Эта команда только для групп")
        return

    cfg = get_chat_cfg(context.bot_data, chat.id)
    cfg["is_banned"] = True

    # Выход из группы
    try:
        await context.bot.leave_chat(chat.id)
        # Сохраняем информацию о выходе
        if "left_groups" not in context.bot_data:
            context.bot_data["left_groups"] = []
        context.bot_data["left_groups"].append({
            "chat_id": chat.id,
            "title": chat.title,
            "banned_at": datetime.now().isoformat()
        })
        await update.message.reply_text(f"✅ Группа {chat.title} заблокирована, бот вышел из группы")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при выходе из группы: {e}")

async def unban_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Разбан группы"""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("⛔ Нет доступа.")
        return

    if not context.args:
        await update.message.reply_text("❌ Укажите ID группы: /unbangroup <group_id>")
        return

    try:
        group_id = int(context.args[0])
        cfg = get_chat_cfg(context.bot_data, group_id)
        if cfg.get("is_banned", False):
            cfg["is_banned"] = False
            await update.message.reply_text(f"✅ Группа {group_id} разблокирована")
        else:
            await update.message.reply_text(f"ℹ️ Группа {group_id} не была забанена")
    except ValueError:
        await update.message.reply_text("❌ Укажите числовой ID группы")

# ---------- Main with auto-restart ----------
def run_once():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(notify_reboot).build()
    
    # Основные команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CommandHandler("draw", draw_cmd))
    app.add_handler(CommandHandler("groupmenu", groupmenu))
    app.add_handler(CommandHandler("mystats", mystats_cmd))  # Персональная статистика
    
    # Админ и спонсор
    app.add_handler(CommandHandler("admin", admin))
    app.add_handler(CommandHandler("sponsor", sponsor_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    
    # Команды банов
    app.add_handler(CommandHandler("ban", ban_user))
    app.add_handler(CommandHandler("unban", unban_user))
    app.add_handler(CommandHandler("bangroup", ban_group))
    app.add_handler(CommandHandler("unbangroup", unban_group))
    
    # Обработчики
    app.add_handler(CallbackQueryHandler(cb))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_msg))
    
    logger.info(f"✅ DeepSeek InlineAdmin {VERSION} запущен. Ожидание событий...")
    app.run_polling()

def main():
    while True:
        try:
            run_once()
        except (NetworkError, TimedOut) as e:
            msg = f"Network error: {type(e).__name__} — {e}"
            logger.warning(f"⚠️ {msg}")
            log_crash(msg)
            time.sleep(5)
            continue
        except Exception as e:
            msg = f"Unhandled error: {type(e).__name__} — {e}"
            logger.error(f"❌ {msg}")
            log_crash(msg)
            time.sleep(10)
            continue

if __name__ == "__main__":
    main()

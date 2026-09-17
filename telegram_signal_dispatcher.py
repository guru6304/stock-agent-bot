"""Telegram Signal Dispatcher for Indian Paper Signals.

Formats and delivers qualifying signals to the configured Telegram destination.
Strictly paper observation. Never claims delivery without positive API confirmation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple
import requests
from dotenv import load_dotenv

from india_market_config import CURRENCY_SYMBOL
from indian_signals import IndianTradeSignal

load_dotenv()
logger = logging.getLogger("telegram-dispatcher")


def format_signal_card(sig: IndianTradeSignal) -> str:
    """Build standardized, rich Telegram Markdown message for an Indian paper signal."""
    direction_emoji = "🟢" if sig.direction == "BUY" else "🔴"

    fno_section = ""
    if sig.fno_details:
        f = sig.fno_details
        fno_section = (
            f"\n📋 *Validated Option Contract:* `{f.get('contract_symbol', '')}`\n"
            f"   • Expiry: `{f.get('expiry', 'N/A')}` | Strike: *{CURRENCY_SYMBOL}{f.get('strike', 0):,.2f}* {f.get('option_type', '')}\n"
            f"   • Token: `{f.get('contract_token', 'N/A')}` | Lot Size: *{f.get('lot_size', 1)}*\n"
            f"   • Underlying Spot: *{CURRENCY_SYMBOL}{f.get('underlying_spot', 0):,.2f}*\n"
        )

    card = (
        f"🚨 *INDIA PAPER SIGNAL — NOT A TRADE*\n"
        f"───────────────────────────────────\n"
        f"🆔 *Signal ID:* `{sig.signal_id}`\n"
        f"📈 *Instrument:* *{sig.symbol}* ({sig.exchange}) | Token: `{sig.token}`\n"
        f"🎯 *Horizon:* `{sig.horizon.value}`\n"
        f"⚖️ *Direction:* {direction_emoji} *{sig.direction}*\n"
        f"🧠 *Strategy:* {sig.strategy_name} (v{sig.strategy_version})\n"
        f"⏰ *Signal Time:* {sig.signal_time_ist}\n"
        f"📡 *Data Source:* {sig.data_source} (`{sig.data_timestamp}`)\n"
        f"💰 *Entry Range:* {CURRENCY_SYMBOL}{sig.entry_range_low:,.2f} – {CURRENCY_SYMBOL}{sig.entry_range_high:,.2f}\n"
        f"🛑 *Strict Stop-Loss:* {CURRENCY_SYMBOL}{sig.stop_loss:,.2f}\n"
        f"🎯 *Target 1:* {CURRENCY_SYMBOL}{sig.target_1:,.2f} | *Target 2:* {CURRENCY_SYMBOL}{sig.target_2:,.2f}\n"
        f"📊 *Risk:Reward:* *1:{sig.risk_reward_ratio:.1f}*\n"
        f"⏳ *Validity:* {sig.validity_period}\n"
        f"{fno_section}"
        f"📝 *Trigger Thesis:* {sig.thesis}\n"
        f"───────────────────────────────────\n"
        f"⚠️ _Signal-only paper observation. Not financial advice. Zero broker orders placed._"
    )
    return card


class TelegramSignalDispatcher:
    """Manages secure delivery of paper signals to Telegram."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.bot_token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.http = session or requests.Session()

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id and not self.bot_token.startswith("your_"))

    def send_signal(self, sig: IndianTradeSignal) -> Tuple[bool, str]:
        """Dispatch a single paper signal to Telegram.

        Returns (success, description).
        """
        if not self.is_configured:
            return False, "Telegram credentials not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing)"

        text = format_signal_card(sig)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            resp = self.http.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    msg_id = data.get("result", {}).get("message_id", "")
                    logger.info("Telegram signal %s delivered successfully (msg_id: %s)", sig.signal_id, msg_id)
                    return True, f"Delivered (msg_id: {msg_id})"
            err = f"Telegram API error HTTP {resp.status_code}: {resp.text}"
            logger.error("Failed to deliver signal %s: %s", sig.signal_id, err)
            return False, err
        except Exception as e:
            err = f"Network exception during Telegram dispatch: {e}"
            logger.error("Failed to deliver signal %s: %s", sig.signal_id, err)
            return False, err

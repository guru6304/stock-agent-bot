"""
Notification Layer — sends trade alerts via Email, Slack, Telegram,
and logs to a local dashboard CSV.
"""

import csv
import json
import logging
import os
import smtplib
import threading
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from dotenv import load_dotenv

from position_sizing import PositionPlan
from signal_engine import TradeAlert

load_dotenv()
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
DASHBOARD_LOG = LOGS_DIR / "dashboard.csv"

# ---------------------------------------------------------------------------
# Daily digest queue — collects all alerts and reports into ONE email per day
# ---------------------------------------------------------------------------

_queue_lock = threading.Lock()
_daily_alert_queue = []   # list of {"alert", "plan", "graded_text", "ts"}
_queued_briefing  = None  # morning briefing plain text
_queued_reports   = []    # list of {"subject", "text", "html"} — all non-alert emails
_digest_sending   = False # True while send_daily_digest() is actually transmitting

# ---------------------------------------------------------------------------
# Telegram batching — collects alerts during a scan, sends ONE message at end
# ---------------------------------------------------------------------------

_telegram_batch_lock = threading.Lock()
_telegram_batch = []  # list of text strings to send as a single message


def queue_telegram(text: str) -> None:
    """Queue a Telegram message for batched delivery instead of sending immediately."""
    with _telegram_batch_lock:
        _telegram_batch.append(text)


def flush_telegram_batch() -> bool:
    """Send all queued Telegram messages as one consolidated message."""
    with _telegram_batch_lock:
        if not _telegram_batch:
            return False
        messages = list(_telegram_batch)
        _telegram_batch.clear()

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id or token.startswith("your_"):
        return False

    combined = "\n\n---\n\n".join(messages)
    # Telegram max message length is 4096 chars
    if len(combined) > 4000:
        combined = combined[:3950] + "\n\n... (truncated)"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": combined}, timeout=15)
        resp.raise_for_status()
        logger.info("Telegram batch sent: %d alerts in 1 message", len(messages))
        return True
    except Exception as e:
        logger.error("Telegram batch send failed: %s", e)
        return False


def queue_alert_for_digest(alert: TradeAlert, plan: PositionPlan, graded_text: str = None) -> None:
    """Queue a trade alert so it ends up in the end-of-day digest instead of a separate email."""
    with _queue_lock:
        _daily_alert_queue.append({
            "alert": alert,
            "plan": plan,
            "graded_text": graded_text,
            "ts": datetime.now(),
        })
    logger.debug("Alert queued for digest: %s %s", alert.direction, alert.ticker)


def set_briefing_for_digest(text: str) -> None:
    """Store the morning briefing text so it appears in the digest email."""
    global _queued_briefing
    with _queue_lock:
        _queued_briefing = text


def _build_digest_html(alerts, briefing_text, eod_text, today_str):
    """Build the full HTML for the daily digest email."""
    buy_alerts  = [a for a in alerts if a["alert"].direction == "BUY"]
    sell_alerts = [a for a in alerts if a["alert"].direction != "BUY"]

    def alert_card(item):
        a = item["alert"]
        p = item["plan"]
        color = "#1b5e20" if a.direction == "BUY" else "#b71c1c"
        badge_bg = "#2e7d32" if a.direction == "BUY" else "#c62828"
        signals = ", ".join(s[0] for s in a.triggered_signals[:5])
        ts = item["ts"].strftime("%H:%M")
        graded = item.get("graded_text") or ""
        grade_line = ""
        for line in graded.splitlines():
            if "Grade:" in line or "grade:" in line.lower():
                grade_line = f"<span style='color:{color};font-weight:bold'>{line.strip()}</span><br>"
                break
        return f"""
        <div style="border:1px solid #ddd;border-radius:8px;margin:8px 0;overflow:hidden;">
          <div style="background:{badge_bg};color:white;padding:10px 14px;display:flex;justify-content:space-between;">
            <strong>{a.direction} {a.ticker}</strong>
            <span>Score: {a.signal_score} &nbsp;|&nbsp; {ts}</span>
          </div>
          <div style="padding:12px 14px;font-size:13px;">
            {grade_line}
            <table style="width:100%;border-collapse:collapse;">
              <tr>
                <td style="padding:3px 6px;color:#555;">Entry</td><td style="padding:3px 6px;"><b>${p.entry_price:,.2f}</b></td>
                <td style="padding:3px 6px;color:#555;">Stop</td><td style="padding:3px 6px;">${p.stop_loss:,.2f}</td>
                <td style="padding:3px 6px;color:#555;">Shares</td><td style="padding:3px 6px;">{p.shares}</td>
              </tr>
              <tr>
                <td style="padding:3px 6px;color:#555;">T1</td><td style="padding:3px 6px;">${p.target_1:,.2f}</td>
                <td style="padding:3px 6px;color:#555;">T2</td><td style="padding:3px 6px;">${p.target_2:,.2f}</td>
                <td style="padding:3px 6px;color:#555;">Max Loss</td><td style="padding:3px 6px;">${p.max_loss:,.2f}</td>
              </tr>
            </table>
            <p style="margin:6px 0 0;color:#666;font-size:12px;">Signals: {signals}</p>
          </div>
        </div>"""

    alert_html = ""
    if buy_alerts:
        alert_html += "<h3 style='color:#2e7d32;margin:16px 0 6px;'>🟢 BUY Signals (%d)</h3>" % len(buy_alerts)
        alert_html += "".join(alert_card(a) for a in buy_alerts)
    if sell_alerts:
        alert_html += "<h3 style='color:#c62828;margin:16px 0 6px;'>🔴 SELL / EXIT Signals (%d)</h3>" % len(sell_alerts)
        alert_html += "".join(alert_card(a) for a in sell_alerts)
    if not alerts:
        alert_html = "<p style='color:#777;font-style:italic;'>No trade signals triggered today.</p>"

    briefing_section = ""
    if briefing_text:
        briefing_section = """
        <h2 style="color:#1565c0;border-bottom:2px solid #1565c0;padding-bottom:6px;">Morning Briefing</h2>
        <pre style="background:#f5f5f5;padding:12px;border-radius:6px;font-size:12px;white-space:pre-wrap;">%s</pre>
        """ % briefing_text[:3000]

    eod_section = ""
    if eod_text:
        eod_section = """
        <h2 style="color:#4a148c;border-bottom:2px solid #4a148c;padding-bottom:6px;">End-of-Day Report</h2>
        <pre style="background:#f3e5f5;padding:12px;border-radius:6px;font-size:12px;white-space:pre-wrap;">%s</pre>
        """ % eod_text[:3000]

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>body{{font-family:Arial,sans-serif;max-width:680px;margin:0 auto;color:#222;}}
h1{{background:linear-gradient(135deg,#0d47a1,#1565c0);color:white;padding:20px;margin:0;border-radius:8px 8px 0 0;}}
.subtitle{{background:#1565c0;color:#cfe2ff;padding:6px 20px;font-size:13px;margin:0;}}
.body{{padding:16px;}}
</style></head>
<body>
<h1>📊 Daily Market Digest</h1>
<p class="subtitle">{today_str} &nbsp;|&nbsp; {len(buy_alerts)} BUY &nbsp;|&nbsp; {len(sell_alerts)} SELL</p>
<div class="body">
{briefing_section}
<h2 style="color:#e65100;border-bottom:2px solid #e65100;padding-bottom:6px;">Today's Trade Signals ({len(alerts)} total)</h2>
{alert_html}
{eod_section}
<hr style="margin:24px 0;border:none;border-top:1px solid #eee;">
<p style="color:#aaa;font-size:11px;text-align:center;">Stock Agent Bot &mdash; automated analysis only, not financial advice.</p>
</div>
</body></html>"""


def send_daily_digest(eod_text: str = None) -> bool:
    """Build and send the ONE daily email with all alerts, briefing, and EOD report."""
    global _queued_briefing, _digest_sending

    host     = os.getenv("SMTP_HOST", "")
    port     = int(os.getenv("SMTP_PORT", "587"))
    user     = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    to_addr  = os.getenv("ALERT_EMAIL_TO", "")

    if not all([host, user, password, to_addr]) or user.startswith("your_"):
        logger.info("Email not configured — skipping daily digest")
        return False

    with _queue_lock:
        alerts  = list(_daily_alert_queue)
        reports = list(_queued_reports)
        briefing_snap = _queued_briefing
        _daily_alert_queue.clear()
        _queued_reports.clear()
        _queued_briefing = None

    today_str = datetime.now().strftime("%A, %B %d, %Y")
    buy_count  = sum(1 for a in alerts if a["alert"].direction == "BUY")
    sell_count = len(alerts) - buy_count
    subject    = f"📊 Daily Digest — {datetime.now().strftime('%b %d')} | {buy_count} BUY, {sell_count} SELL"

    html = _build_digest_html(alerts, briefing_snap, eod_text, today_str)

    # Build reports section (sell alerts, news, earnings, etc. queued during the day)
    reports_html = ""
    reports_text = ""
    if reports:
        reports_text = "\n\nOTHER ALERTS (%d)\n%s\n" % (len(reports), "-" * 30)
        for r in reports:
            reports_text += "\n[%s]\n%s\n" % (r["subject"], r["text"][:800])
        reports_html = "<h2 style='color:#37474f;border-bottom:2px solid #37474f;padding-bottom:6px;'>Other Alerts (%d)</h2>" % len(reports)
        for r in reports:
            rhtml = r.get("html") or ("<pre style='white-space:pre-wrap;font-size:12px;'>%s</pre>" % r["text"][:1000])
            reports_html += "<details style='margin:8px 0;border:1px solid #ddd;border-radius:6px;'><summary style='padding:10px;cursor:pointer;background:#f5f5f5;'><strong>%s</strong></summary><div style='padding:12px;'>%s</div></details>" % (r["subject"], rhtml)

    # Plain-text fallback
    lines = ["DAILY DIGEST — %s" % today_str, "=" * 50, ""]
    if briefing_snap:
        lines += ["MORNING BRIEFING", "-" * 30, briefing_snap[:1500], ""]
    lines += ["TRADE SIGNALS (%d total)" % len(alerts), "-" * 30]
    for item in alerts:
        a, p = item["alert"], item["plan"]
        lines.append("%s %s  score:%d  entry:$%.2f  stop:$%.2f  T1:$%.2f" % (
            a.direction, a.ticker, a.signal_score, p.entry_price, p.stop_loss, p.target_1))
    if eod_text:
        lines += ["", "END-OF-DAY REPORT", "-" * 30, eod_text[:1500]]
    lines.append(reports_text)
    plain = "\n".join(lines)

    # Inject reports section into HTML
    full_html = html.replace("</body>", reports_html + "</body>")

    _digest_sending = True
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = user
        msg["To"]      = to_addr
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(full_html, "html"))
        for attempt in range(3):
            try:
                with smtplib.SMTP(host, port) as server:
                    server.starttls()
                    server.login(user, password)
                    server.sendmail(user, [to_addr], msg.as_string())
                logger.info("Daily digest sent: %d alerts, %d reports", len(alerts), len(reports))
                return True
            except (smtplib.SMTPException, OSError) as smtp_err:
                logger.warning("Digest SMTP attempt %d/3 failed: %s", attempt + 1, smtp_err)
                if attempt < 2:
                    time.sleep(5)
        logger.error("Daily digest failed after 3 SMTP attempts")
        return False
    except Exception as e:
        logger.error("Daily digest send failed: %s", e)
        return False
    finally:
        _digest_sending = False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_alert_text(alert: TradeAlert, plan: PositionPlan) -> str:
    """Build a human-readable alert message."""
    signals_str = ", ".join(f"{s[0]} (+{s[1]})" for s in alert.triggered_signals)
    fund_details = ""
    if alert.fundamental:
        fund_details = (
            f"\n  Fundamental score: {alert.fundamental.fundamental_score}/6"
            f"\n  Details: {', '.join(alert.fundamental.score_details) or 'N/A'}"
        )

    tech_details = ""
    if alert.technical:
        t = alert.technical
        tech_details = (
            f"\n  EMA9: {t.ema9:.2f}  |  EMA21: {t.ema21:.2f}  |  SMA200: {t.sma200:.2f}"
            f"\n  RSI: {t.rsi:.1f}  |  MACD: {t.macd_value:.2f}  |  BB: [{t.bb_lower:.2f}–{t.bb_upper:.2f}]"
            f"\n  Patterns: {', '.join(t.pattern_details) or 'none'}"
        )

    text = (
        f"{'=' * 55}\n"
        f"  TRADE ALERT: {alert.direction} {alert.ticker}\n"
        f"{'=' * 55}\n"
        f"  Signal score: {alert.signal_score} (threshold met)\n"
        f"  Signals: {signals_str}\n"
        f"{tech_details}"
        f"{fund_details}\n"
        f"\n"
        f"  POSITION PLAN\n"
        f"  {'─' * 40}\n"
        f"  Entry price:    ${plan.entry_price:,.2f}\n"
        f"  Stop-loss:      ${plan.stop_loss:,.2f}\n"
        f"  Risk/share:     ${plan.risk_per_share:,.2f}\n"
        f"  Shares:         {plan.shares}\n"
        f"  Position value: ${plan.position_value:,.2f}\n"
        f"  Max loss:       ${plan.max_loss:,.2f}\n"
        f"\n"
        f"  TARGETS\n"
        f"  T1 (1.5:1 R/R): ${plan.target_1:,.2f}\n"
        f"  T2 (3.0:1 R/R): ${plan.target_2:,.2f}\n"
        f"  T3 (Fib ext):   ${plan.target_3:,.2f}\n"
        f"{'=' * 55}\n"
    )
    return text


def format_alert_html(alert: TradeAlert, plan: PositionPlan) -> str:
    """Build an HTML version for email."""
    color = "#2e7d32" if alert.direction == "BUY" else "#c62828"
    signals_str = ", ".join(f"{s[0]} (+{s[1]})" for s in alert.triggered_signals)

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto;">
      <div style="background: {color}; color: white; padding: 16px; border-radius: 8px 8px 0 0;">
        <h2 style="margin: 0;">{alert.direction} {alert.ticker}</h2>
        <p style="margin: 4px 0 0;">Signal score: {alert.signal_score}</p>
      </div>
      <div style="border: 1px solid #ddd; padding: 16px; border-radius: 0 0 8px 8px;">
        <p><strong>Signals:</strong> {signals_str}</p>
        <table style="width: 100%; border-collapse: collapse;">
          <tr><td style="padding: 4px 0;"><strong>Entry</strong></td><td>${plan.entry_price:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>Stop-loss</strong></td><td>${plan.stop_loss:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>Shares</strong></td><td>{plan.shares}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>Position</strong></td><td>${plan.position_value:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>Max loss</strong></td><td>${plan.max_loss:,.2f}</td></tr>
          <tr style="border-top: 1px solid #eee;"><td style="padding: 4px 0;"><strong>T1 (1.5:1)</strong></td><td>${plan.target_1:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>T2 (3:1)</strong></td><td>${plan.target_2:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>T3 (Fib)</strong></td><td>${plan.target_3:,.2f}</td></tr>
        </table>
      </div>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# Email notification
# ---------------------------------------------------------------------------

def send_email(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert via SMTP email."""
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    to_addr = os.getenv("ALERT_EMAIL_TO", "")

    if not all([host, user, password, to_addr]) or user.startswith("your_"):
        logger.info("Email not configured — skipping")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Trade Alert: {alert.direction} {alert.ticker} (score {alert.signal_score})"
        msg["From"] = user
        msg["To"] = to_addr

        msg.attach(MIMEText(format_alert_text(alert, plan), "plain"))
        msg.attach(MIMEText(format_alert_html(alert, plan), "html"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())

        logger.info("Email sent for %s to %s", alert.ticker, to_addr)
        return True
    except Exception as e:
        logger.error("Email send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Slack notification
# ---------------------------------------------------------------------------

def send_slack(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert to Slack via webhook."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url or webhook_url.startswith("https://hooks.slack.com/services/YOUR"):
        logger.info("Slack webhook not configured — skipping")
        return False

    text = format_alert_text(alert, plan)
    payload = {
        "channel": os.getenv("SLACK_CHANNEL", "#stock-alerts"),
        "username": "Stock Agent",
        "text": f"```{text}```",
        "icon_emoji": ":chart_with_upwards_trend:",
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Slack notification sent for %s", alert.ticker)
        return True
    except Exception as e:
        logger.error("Slack send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

def send_telegram(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert to Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id or token.startswith("your_"):
        logger.info("Telegram not configured — skipping")
        return False

    text = format_alert_text(alert, plan)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram notification sent for %s", alert.ticker)
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def send_telegram_text(text: str) -> bool:
    """Send a plain-text integration message, e.g. an F&O trade card or EOD report."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id or token.startswith("your_"):
        logger.info("Telegram not configured — skipping")
        return False

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": str(text)[:4000]},
            timeout=15,
        )
        response.raise_for_status()
        logger.info("Telegram text notification sent")
        return True
    except requests.RequestException as exc:
        logger.error("Telegram text send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Discord notification
# ---------------------------------------------------------------------------

def send_discord(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert to Discord via webhook."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url or "discord.com/api/webhooks" not in webhook_url:
        logger.info("Discord webhook not configured — skipping")
        return False

    color = 0x2E7D32 if alert.direction == "BUY" else 0xC62828
    signals_str = ", ".join(s[0] for s in alert.triggered_signals)

    embed = {
        "title": "%s %s — Score %d" % (alert.direction, alert.ticker, alert.signal_score),
        "color": color,
        "fields": [
            {"name": "Signals", "value": signals_str, "inline": False},
            {"name": "Entry", "value": "$%.2f" % plan.entry_price, "inline": True},
            {"name": "Stop", "value": "$%.2f" % plan.stop_loss, "inline": True},
            {"name": "Shares", "value": str(plan.shares), "inline": True},
            {"name": "T1 (1.5:1)", "value": "$%.2f" % plan.target_1, "inline": True},
            {"name": "T2 (3:1)", "value": "$%.2f" % plan.target_2, "inline": True},
            {"name": "Max Loss", "value": "$%.2f" % plan.max_loss, "inline": True},
        ],
        "footer": {"text": "Stock Agent"},
        "timestamp": datetime.utcnow().isoformat(),
    }

    payload = {
        "username": "Stock Agent",
        "embeds": [embed],
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord notification sent for %s", alert.ticker)
        return True
    except Exception as e:
        logger.error("Discord send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Generic webhook notification
# ---------------------------------------------------------------------------

def send_sms(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert via SMS using email-to-SMS gateway."""
    gateway = os.getenv("SMS_GATEWAY", "")
    host = os.getenv("SMTP_HOST", "")
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")

    if not gateway or not all([host, user, password]) or user.startswith("your_") or password.startswith("your_"):
        logger.info("SMS not configured — skipping")
        return False

    # SMS is short — compact format
    signals = ", ".join(s[0] for s in alert.triggered_signals[:3])
    text = (
        f"{alert.direction} {alert.ticker} (score {alert.signal_score})\n"
        f"Entry: ${plan.entry_price:,.2f}\n"
        f"Stop: ${plan.stop_loss:,.2f}\n"
        f"Shares: {plan.shares}\n"
        f"T1: ${plan.target_1:,.2f}\n"
        f"Signals: {signals}"
    )

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        msg = MIMEText(text)
        msg["From"] = user
        msg["To"] = gateway
        msg["Subject"] = f"{alert.direction} {alert.ticker}"

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [gateway], msg.as_string())

        logger.info("SMS sent for %s to %s", alert.ticker, gateway)
        return True
    except Exception as e:
        logger.error("SMS send failed: %s", e)
        return False


def send_sms_text(text: str, subject: str = "Stock Agent") -> bool:
    """Send a plain text SMS message (for sell alerts, briefings, etc.)."""
    gateway = os.getenv("SMS_GATEWAY", "")
    host = os.getenv("SMTP_HOST", "")
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")

    if not gateway or not all([host, user, password]) or user.startswith("your_") or password.startswith("your_"):
        return False

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        msg = MIMEText(text[:160])  # SMS limit
        msg["From"] = user
        msg["To"] = gateway
        msg["Subject"] = subject

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [gateway], msg.as_string())

        logger.info("SMS text sent: %s", subject)
        return True
    except Exception as e:
        logger.error("SMS text send failed: %s", e)
        return False


def send_email_text(text: str, subject: str = "Stock Agent", html: str = None) -> bool:
    """
    Queue content for the daily digest instead of sending immediately.

    All modules (briefings, sell alerts, news, earnings, sector rotation,
    upgrades, volume spikes, weekly report) call this. By routing everything
    through the queue, the user receives ONE consolidated email per day at
    4:25 PM instead of 20+ individual emails.

    The only exception is when called from inside send_daily_digest() itself
    (flagged by _digest_sending=True), which does the actual SMTP send.
    """
    global _digest_sending

    if _digest_sending:
        # We are inside send_daily_digest() — actually transmit this one email
        return _smtp_send_raw(text, subject, html)

    # Otherwise: queue for the end-of-day digest
    with _queue_lock:
        _queued_reports.append({"subject": subject, "text": text, "html": html})
    logger.debug("Email queued for digest: %s", subject)
    return True


def _smtp_send_raw(text: str, subject: str, html: str = None) -> bool:
    """Internal: actually send an email via SMTP (only called from send_daily_digest)."""
    host     = os.getenv("SMTP_HOST", "")
    port     = int(os.getenv("SMTP_PORT", "587"))
    user     = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    to_addr  = os.getenv("ALERT_EMAIL_TO", "")

    if not all([host, user, password, to_addr]) or user.startswith("your_") or password.startswith("your_"):
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = user
        msg["To"]      = to_addr
        msg.attach(MIMEText(text, "plain"))
        if html:
            msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())

        logger.info("Email sent: %s", subject)
        return True
    except Exception as e:
        logger.error("Email send failed: %s", e)
        return False


def send_webhook(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert to a generic webhook URL as JSON.

    The payload includes all alert and plan details so any downstream
    system (Zapier, IFTTT, n8n, custom server) can consume it.
    """
    webhook_url = os.getenv("CUSTOM_WEBHOOK_URL", "")
    if not webhook_url:
        return False

    signals_str = [s[0] for s in alert.triggered_signals]
    fund_score = alert.fundamental.fundamental_score if alert.fundamental else 0

    payload = {
        "event": "trade_alert",
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": alert.ticker,
        "direction": alert.direction,
        "signal_score": alert.signal_score,
        "triggered_signals": signals_str,
        "fundamental_score": fund_score,
        "plan": {
            "entry_price": plan.entry_price,
            "stop_loss": plan.stop_loss,
            "risk_per_share": plan.risk_per_share,
            "shares": plan.shares,
            "position_value": plan.position_value,
            "max_loss": plan.max_loss,
            "target_1": plan.target_1,
            "target_2": plan.target_2,
            "target_3": plan.target_3,
        },
    }

    # Support custom headers (e.g., API keys)
    headers = {"Content-Type": "application/json"}
    auth_header = os.getenv("CUSTOM_WEBHOOK_AUTH", "")
    if auth_header:
        headers["Authorization"] = auth_header

    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        logger.info("Webhook notification sent for %s to %s", alert.ticker, webhook_url)
        return True
    except Exception as e:
        logger.error("Webhook send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Dashboard log (CSV)
# ---------------------------------------------------------------------------

def log_to_dashboard(alert: TradeAlert, plan: PositionPlan) -> None:
    """Append alert to the local dashboard CSV log and optionally to SQLite."""
    signals_str = "|".join(s[0] for s in alert.triggered_signals)
    fund_score = alert.fundamental.fundamental_score if alert.fundamental else 0
    news_sent = alert.fundamental.news_sentiment_score if alert.fundamental else 0

    # Log to SQLite if enabled
    if os.getenv("USE_SQLITE", "0") == "1":
        try:
            import database as db
            db.insert_alert(
                ticker=alert.ticker, direction=alert.direction,
                signal_score=alert.signal_score, signals=signals_str,
                entry_price=plan.entry_price, stop_loss=plan.stop_loss,
                shares=plan.shares, position_value=plan.position_value,
                max_loss=plan.max_loss, target_1=plan.target_1,
                target_2=plan.target_2, target_3=plan.target_3,
                fundamental_score=fund_score, news_sentiment=news_sent,
            )
        except Exception as e:
            logger.debug("DB alert insert fallback: %s", e)

    # Always write CSV too (backward compat)
    file_exists = DASHBOARD_LOG.exists()
    row = {
        "timestamp": datetime.now().isoformat(),
        "ticker": alert.ticker,
        "direction": alert.direction,
        "signal_score": alert.signal_score,
        "signals": signals_str,
        "entry_price": plan.entry_price,
        "stop_loss": plan.stop_loss,
        "shares": plan.shares,
        "position_value": plan.position_value,
        "max_loss": plan.max_loss,
        "target_1": plan.target_1,
        "target_2": plan.target_2,
        "target_3": plan.target_3,
        "fundamental_score": fund_score,
        "news_sentiment": news_sent,
    }

    fieldnames = list(row.keys())
    with open(DASHBOARD_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Dashboard log updated for %s", alert.ticker)


# ---------------------------------------------------------------------------
# Send all notifications
# ---------------------------------------------------------------------------

def notify(alert: TradeAlert, plan: PositionPlan) -> None:
    """Dispatch alert through all configured channels with grade + options."""
    # Always log to dashboard
    log_to_dashboard(alert, plan)

    # Grade the trade and generate options suggestion
    graded_text = None
    try:
        import trade_grader
        import options_analyzer

        grade_info = trade_grader.grade_trade(alert, plan)
        options_suggestion = None
        try:
            options_suggestion = options_analyzer.analyze_ticker_options(
                alert.ticker, alert.direction, alert.signal_score
            )
        except Exception as e:
            logger.debug("Options analysis failed for %s: %s", alert.ticker, e)

        graded_text = trade_grader.format_graded_alert(
            alert, plan, grade_info, options_suggestion
        )
    except Exception as e:
        logger.debug("Trade grading failed for %s: %s", alert.ticker, e)

    # Print to console (graded version if available)
    if graded_text:
        print(graded_text)
    else:
        print(format_alert_text(alert, plan))

    # Queue for batched Telegram delivery (one message per scan, not per alert)
    queue_telegram(graded_text if graded_text else format_alert_text(alert, plan))

    # Queue alert for end-of-day digest instead of sending an individual email
    queue_alert_for_digest(alert, plan, graded_text)

    # Send through other configured channels (standard format — no email, handled by digest)
    # SMS disabled — email-to-SMS gateway is unreliable and causes bounce spam
    send_slack(alert, plan)
    send_discord(alert, plan)
    send_webhook(alert, plan)

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def synthesize_with_gemini(symbol: str, signal_data: dict) -> str:
    """Uses Gemini 2.5 Flash to evaluate equity setups before Telegram dispatch."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return ""

    try:
        client = genai.Client(api_key=gemini_key)
        prompt = f"""
        You are an expert Equity Swing and Intraday Trader.
        Evaluate this algorithmic signal:
        - Symbol: {symbol}
        - Metrics & Setup: {signal_data}

        Provide a 2-sentence actionable verdict for Telegram:
        1. State clearly: [HIGH CONVICTION / SPECULATIVE / AVOID]
        2. Key risk factor (e.g. invalidation level or index market regime drag).
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return f"\n\n🤖 *Gemini AI Verdict:*\n{response.text.strip()}"
    except Exception as e:
        return f"\n\n*(Gemini synthesis unavailable: {e})*"

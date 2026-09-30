"""Real-time NSE Stock Recommendation Engine: scan, briefing, eod.

High-performance intraday & F&O momentum scanner with deduplication,
Call/Put options contract matching, and automated EOD P&L audit.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

try:
    from SmartApi import SmartConnect
    import pyotp
except ImportError:
    SmartConnect = pyotp = None

try:
    from google import genai
except ImportError:
    genai = None

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("nse-engine")
HTTP = requests.Session()
HTTP.headers.update({"User-Agent": "Mozilla/5.0 (NSE Stock Recommendation Engine)"})

NSE_50_CSV = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
NSE_NEXT50_CSV = "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv"
SIGNAL_LOG = Path(os.getenv("SIGNAL_LOG", "logs/nse_signals.jsonl"))

def env(k: str) -> str:
    return os.getenv(k, "").strip()

def symbol(x: str) -> str:
    return x.upper().strip().replace(".NS", "").replace("-EQ", "")

@dataclass
class Metrics:
    symbol: str
    price: float
    high: float
    low: float
    sma20: float
    ema20: float
    ema50: float
    ema200: float
    vol_ratio: float
    high3: float
    low3: float
    high20: float
    low20: float
    atr: float
    vwap: Optional[float]
    orb_high: Optional[float]
    orb_low: Optional[float]
    rsi: float
    source: str
    quality: bool

@dataclass
class Signal:
    symbol: str
    style: str
    action: str  # "BUY", "SHORT", "ACCUMULATE"
    entry_low: float
    entry_high: float
    stop: float
    target1: float
    target2: float
    rr: float
    thesis: str
    source: str
    fno_info: Optional[str] = None

# High-liquidity F&O momentum leaders across key NSE sectors
CORE_FNO_MOVERS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC",
    "KOTAKBANK", "LT", "AXISBANK", "TATAMOTORS", "MARUTI", "SUNPHARMA", "TITAN",
    "BAJFINANCE", "TATASTEEL", "NTPC", "POWERGRID", "M&M", "ADANIENT", "ADANIPORTS",
    "COALINDIA", "ONGC", "TRENT", "BEL", "HAL", "DIXON", "BHEL", "CANBK", "POLYCAB",
    "ZOMATO", "SUZLON", "FEDERALBNK", "PFC", "REC", "TATAPOWER", "ASHOKLEY", "DLF",
    "JINDALSTEL", "HINDALCO", "VEDL", "CHOLAFIN", "INDUSINDBK", "PERSISTENT", "COFORGE",
    "APOLLOHOSP", "SIEMENS", "ABB", "CUMMINSIND", "VOLTAS", "TATACHEM", "JUBLFOOD",
    "MUTHOOTFIN", "SHRIRAMFIN", "AUBANK", "BANDHANBNK", "IDFCFIRSTB", "SAIL", "NMDC",
    "NATIONALUM", "IOC", "BPCL", "HINDPETRO", "GAIL", "IGL", "MGL", "PETRONET",
    "DIVISLAB", "CIPLA", "DRREDDY", "LUPIN", "AUROPHARMA", "TORNTPHARM", "BIOCON",
    "HEROMOTOCO", "BAJAJ-AUTO", "EICHERMOT", "TVSMOTOR", "BALKRISIND", "MOTHERSON",
    "NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP", "MARICO", "COLPAL", "HINDUNILVR",
    "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS", "LTTS", "KPITTECH",
    "ULTRACEMCO", "GRASIM", "AMBUJACEM", "ACC", "DALBHARAT", "INDIGO", "IRCTC",
]

def watchlist() -> list[str]:
    """Build an expanded, highly liquid universe covering Nifty 100 and active F&O leaders."""
    if env("WATCHLIST"):
        return list(dict.fromkeys(symbol(x) for x in env("WATCHLIST").split(",") if symbol(x)))

    seen = set()
    out = []

    # 1. Include core liquid F&O stocks first
    for s in CORE_FNO_MOVERS:
        s_clean = symbol(s)
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            out.append(s_clean)

    # 2. Enrich from live NSE Nifty 50 and Nifty Next 50 lists
    for url in [NSE_50_CSV, NSE_NEXT50_CSV]:
        try:
            r = HTTP.get(url, timeout=8)
            if r.status_code == 200:
                for row in csv.DictReader(io.StringIO(r.text)):
                    sym = symbol(row.get("Symbol", ""))
                    if sym and sym not in seen:
                        seen.add(sym)
                        out.append(sym)
        except Exception as e:
            LOG.debug("NSE constituent list fetch error: %s", e)

    limit = int(os.getenv("MAX_SCAN_UNIVERSE", "120"))
    return out[:limit]

class Angel:
    """SmartAPI authentication and LTP source; failures gracefully fall back to yfinance."""
    def __init__(self):
        self.api = None
        self.tokens = {}
        required = ("ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_PIN", "ANGEL_TOTP_KEY")
        if not all(env(x) for x in required):
            return
        if not SmartConnect or not pyotp:
            LOG.warning("Angel credentials found but smartapi-python/pyotp are not installed")
            return
        try:
            self.api = SmartConnect(api_key=env("ANGEL_API_KEY"))
            otp = pyotp.TOTP(env("ANGEL_TOTP_KEY")).now()
            session = self.api.generateSession(env("ANGEL_CLIENT_CODE"), env("ANGEL_PIN"), otp)
            if not session or not session.get("status"):
                raise RuntimeError((session or {}).get("message", "login rejected"))
            LOG.info("Angel One session authenticated")
            try:
                scrip_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
                data = HTTP.get(scrip_url, timeout=25).json()
                self.tokens = {
                    str(x.get("symbol", "")).replace("-EQ", ""): str(x.get("token", ""))
                    for x in data
                    if x.get("exch_seg") == "NSE" and str(x.get("symbol", "")).endswith("-EQ")
                }
                LOG.info("Angel One token map loaded (%d equity tokens)", len(self.tokens))
            except Exception as te:
                LOG.warning("Could not load Angel token map: %s", te)
        except Exception as e:
            self.api = None
            LOG.warning("Angel unavailable; falling back to yfinance NSE: %s", e)

    def quote(self, s: str) -> Optional[float]:
        if not self.api or s not in self.tokens:
            return None
        try:
            q = self.api.ltpData("NSE", s + "-EQ", self.tokens[s])
            value = (q or {}).get("data", {}).get("ltp")
            return float(value) if value else None
        except Exception:
            return None

def yf_history(s: str, period: str, interval="1d") -> pd.DataFrame:
    try:
        d = yf.download(s + ".NS", period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return d.dropna(how="all")
    except Exception as e:
        LOG.debug("NSE fallback failed for %s: %s", s, e)
        return pd.DataFrame()

def get_metrics(s: str, angel: Angel) -> Optional[Metrics]:
    d = yf_history(s, "1y")
    if len(d) < 50 or not {"Close", "High", "Low", "Volume"}.issubset(d):
        return None

    c, h, l, v = (d[x].astype(float) for x in ("Close", "High", "Low", "Volume"))
    live = angel.quote(s)
    price = live or float(c.iloc[-1])

    e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
    e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
    e200 = float(c.ewm(span=min(200, len(c)), adjust=False).mean().iloc[-1])
    sma20 = float(c.tail(20).mean())

    tr = pd.concat((h - l, (h - c.shift()).abs(), (l - c.shift()).abs()), axis=1).max(axis=1)
    atr = float(tr.tail(14).mean()) if len(tr) >= 14 else float(h.iloc[-1] - l.iloc[-1])

    # 20d average volume
    avg_vol_20 = max(float(v.tail(21).iloc[:-1].mean()), 1.0)
    vol_ratio = float(v.iloc[-1] / avg_vol_20)

    # Relative strength (RSI 14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if len(c) >= 15 else 50.0

    # 3-day and 20-day high / low
    high3 = float(h.tail(4).iloc[:-1].max()) if len(h) >= 4 else float(h.max())
    low3 = float(l.tail(4).iloc[:-1].min()) if len(l) >= 4 else float(l.min())
    high20 = float(h.tail(21).iloc[:-1].max()) if len(h) >= 21 else float(h.max())
    low20 = float(l.tail(21).iloc[:-1].min()) if len(l) >= 21 else float(l.min())

    # Intraday 5m data for VWAP and Opening Range (ORB)
    intra = yf_history(s, "5d", "5m")
    vw = orb_high = orb_low = None
    if not intra.empty and {"High", "Low", "Close", "Volume"}.issubset(intra):
        typical = (intra.High + intra.Low + intra.Close) / 3
        denom = intra.Volume.sum()
        vw = float((typical * intra.Volume).sum() / denom) if denom else None
        orb_high = float(intra.High.tail(15).max())
        orb_low = float(intra.Low.tail(15).min())

    quality = price > e200 and (len(c) < 252 or float(c.pct_change(min(252, len(c)-1), fill_method=None).iloc[-1]) > -0.20)

    return Metrics(
        symbol=s,
        price=price,
        high=float(h.iloc[-1]),
        low=float(l.iloc[-1]),
        sma20=sma20,
        ema20=e20,
        ema50=e50,
        ema200=e200,
        vol_ratio=vol_ratio,
        high3=high3,
        low3=low3,
        high20=high20,
        low20=low20,
        atr=atr,
        vwap=vw,
        orb_high=orb_high,
        orb_low=orb_low,
        rsi=rsi,
        source="Angel One SmartAPI" if live else "yfinance NSE",
        quality=quality,
    )

def get_fno_option_play(symbol: str, price: float, is_bullish: bool) -> Optional[str]:
    """Calculate matching ATM/OTM Option strike with estimated targets for F&O stocks."""
    if symbol not in CORE_FNO_MOVERS:
        return None
    if price < 100: interval = 2.5
    elif price < 250: interval = 5.0
    elif price < 500: interval = 10.0
    elif price < 1000: interval = 20.0
    elif price < 2500: interval = 50.0
    elif price < 5000: interval = 100.0
    else: interval = 200.0

    strike = round(price / interval) * interval
    opt_type = "CE" if is_bullish else "PE"
    strike_str = f"{int(strike)}" if strike.is_integer() else f"{strike:.1f}"

    est_prem = round(price * 0.025, 1)
    tgt1 = round(est_prem * 1.5, 1)
    tgt2 = round(est_prem * 2.2, 1)
    sl = round(est_prem * 0.65, 1)

    return f"`{symbol} {strike_str} {opt_type}` | Est. Premium: ~₹{est_prem} | TGT: ₹{tgt1}/₹{tgt2} | SL: ₹{sl}"

def classify(m: Metrics) -> Optional[Signal]:
    """Evaluate both Bullish (BUY/CE) and Bearish (SHORT/PE) setups across horizons."""
    p = m.price
    atr = max(m.atr, p * 0.003)
    style = action = thesis = None
    fno_info = None

    # 1. INTRADAY BULLISH BREAKOUT
    if m.vwap and m.orb_high and p > m.vwap and p >= m.orb_high * 0.998 and m.vol_ratio >= 1.25 and m.rsi >= 48:
        style, action = "INTRADAY", "BUY"
        stop = min(m.vwap, p * 0.992)
        thesis = f"VWAP reclaim holding above morning opening range with {m.vol_ratio:.1f}x volume surge (RSI: {m.rsi:.0f})."
        fno_info = get_fno_option_play(m.symbol, p, is_bullish=True)

    # 2. INTRADAY BEARISH BREAKDOWN (SHORT)
    elif m.vwap and m.orb_low and p < m.vwap and p <= m.orb_low * 1.002 and m.vol_ratio >= 1.25 and m.rsi <= 52:
        style, action = "INTRADAY", "SHORT"
        stop = max(m.vwap, p * 1.008)
        thesis = f"Rejection below VWAP and breakdown of morning opening low with {m.vol_ratio:.1f}x volume surge (RSI: {m.rsi:.0f})."
        fno_info = get_fno_option_play(m.symbol, p, is_bullish=False)

    # 3. SHORT-TERM BULLISH BREAKOUT
    elif p > m.high3 and p > m.ema20 and m.vol_ratio >= 1.30:
        style, action = "SHORT-TERM", "BUY"
        stop = min(m.ema20, p - 1.2 * atr)
        thesis = f"Multi-day resistance breakout confirmed by {m.vol_ratio:.1f}x relative volume above rising 20 EMA."
        fno_info = get_fno_option_play(m.symbol, p, is_bullish=True)

    # 4. SHORT-TERM BEARISH BREAKDOWN
    elif p < m.low3 and p < m.ema20 and m.vol_ratio >= 1.30:
        style, action = "SHORT-TERM", "SHORT"
        stop = max(m.ema20, p + 1.2 * atr)
        thesis = f"Multi-day support breakdown below 20 EMA with {m.vol_ratio:.1f}x selling volume confirmation."
        fno_info = get_fno_option_play(m.symbol, p, is_bullish=False)

    # 5. SWING TREND CONTINUATION
    elif p > m.high20 and m.ema20 > m.ema50 and m.vol_ratio >= 1.15:
        style, action = "SWING", "BUY"
        stop = min(m.ema20, p - 1.5 * atr)
        thesis = "Consolidation breakout aligning with institutional 20/50 EMA bullish trend alignment."
        fno_info = get_fno_option_play(m.symbol, p, is_bullish=True)

    # 6. LONG-TERM VALUE ACCUMULATE
    elif m.quality and abs(p / m.ema200 - 1) <= 0.035 and m.ema20 > m.ema50:
        style, action = "LONG-TERM", "ACCUMULATE"
        stop = p - 2.5 * atr
        thesis = "Price retesting rising 200-day EMA support with strong long-term trend anchor."

    else:
        return None

    if "BUY" in action or action == "ACCUMULATE":
        risk = p - stop
        if risk <= 0:
            return None
        t1, t2 = p + 2.0 * risk, p + 3.5 * risk
        if style == "LONG-TERM":
            t1, t2 = max(t1, p * 1.15), max(t2, p * 1.25)
        rr = (t1 - p) / risk
        return Signal(
            symbol=m.symbol,
            style=style,
            action=action,
            entry_low=round(p * 0.997, 2),
            entry_high=round(p * 1.003, 2),
            stop=round(stop, 2),
            target1=round(t1, 2),
            target2=round(t2, 2),
            rr=round(rr, 1),
            thesis=thesis,
            source=m.source,
            fno_info=fno_info,
        )
    else:  # SHORT / SELL
        risk = stop - p
        if risk <= 0:
            return None
        t1, t2 = p - 2.0 * risk, p - 3.5 * risk
        rr = (p - t1) / risk
        return Signal(
            symbol=m.symbol,
            style=style,
            action=action,
            entry_low=round(p * 0.997, 2),
            entry_high=round(p * 1.003, 2),
            stop=round(stop, 2),
            target1=round(t1, 2),
            target2=round(t2, 2),
            rr=round(rr, 1),
            thesis=thesis,
            source=m.source,
            fno_info=fno_info,
        )

def ai_thesis(s: Signal, m: Metrics) -> str:
    """Gemini synthesizes the card trigger from technical inputs if configured."""
    if not env("GEMINI_API_KEY") or not genai:
        return s.thesis
    prompt = (
        "You are the synthesis step for a concrete NSE trade card. Return exactly one factual "
        "technical-trigger sentence, no greeting, disclaimer, or prediction. The card's fixed values are: "
        f"Ticker={m.symbol}; Trading Style={s.style}; Action={s.action}; Entry=₹{s.entry_low:.2f}-₹{s.entry_high:.2f}; "
        f"Stop=₹{s.stop:.2f}; Target1=₹{s.target1:.2f}; Target2=₹{s.target2:.2f}; R:R=1:{s.rr:.1f}. "
        f"Indicators: Price={m.price:.2f}; SMA20={m.sma20:.2f}; EMA20={m.ema20:.2f}; EMA50={m.ema50:.2f}; "
        f"EMA200={m.ema200:.2f}; Volume/20d={m.vol_ratio:.2f}; DayRange={m.low:.2f}-{m.high:.2f}. Base trigger={s.thesis}"
    )
    try:
        client = genai.Client(api_key=env("GEMINI_API_KEY"))
        res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = re.sub(r"\s+", " ", res.text.strip())
        return text if text and len(text) <= 240 else s.thesis
    except Exception as e:
        LOG.debug("Gemini synthesis skipped for %s: %s", m.symbol, e)
        return s.thesis

def card(s: Signal) -> str:
    direction_icon = "🟢" if "BUY" in s.action or s.action == "ACCUMULATE" else "🔴"
    fno_line = f"\n🎯 *F&O Option Play:* {s.fno_info}" if s.fno_info else ""
    return (
        f"*{s.symbol} — {s.style}*\n"
        f"Action: {direction_icon} *{s.action}*\n"
        f"Entry Zone: ₹{s.entry_low:,.2f} – ₹{s.entry_high:,.2f}\n"
        f"Strict Stop-Loss: ₹{s.stop:,.2f}\n"
        f"Target 1: ₹{s.target1:,.2f} | Target 2: ₹{s.target2:,.2f}\n"
        f"Risk:Reward: *1:{s.rr:.1f}*"
        f"{fno_line}\n"
        f"Trigger: {s.thesis}\n"
        f"Data: {s.source}"
    )

def telegram(text: str) -> bool:
    if not env("TELEGRAM_BOT_TOKEN") or not env("TELEGRAM_CHAT_ID"):
        LOG.error("Telegram credentials missing")
        return False
    try:
        r = HTTP.post(
            f"https://api.telegram.org/bot{env('TELEGRAM_BOT_TOKEN')}/sendMessage",
            json={
                "chat_id": env("TELEGRAM_CHAT_ID"),
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
            timeout=15,
        )
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        LOG.error("Telegram dispatch failed: %s", e)
        return False

def get_today_emitted_keys() -> set[str]:
    """Return set of 'SYMBOL:ACTION' signals already alerted today."""
    today = datetime.now().date().isoformat()
    keys = set()
    if SIGNAL_LOG.exists():
        for line in SIGNAL_LOG.read_text(encoding="utf8").splitlines():
            try:
                rec = json.loads(line)
                created = str(rec.get("created_at", ""))
                if created.startswith(today):
                    keys.add(f"{rec.get('symbol')}:{rec.get('action')}")
            except Exception:
                pass
    return keys

def log_signals(ss: list[Signal]):
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SIGNAL_LOG.open("a", encoding="utf8") as f:
        for s in ss:
            f.write(json.dumps({"created_at": datetime.now().astimezone().isoformat(), **asdict(s)}) + "\n")

def scan(force: bool = False):
    """Scan expanded liquid universe, apply strict deduplication, and emit only FRESH signals."""
    LOG.info("Executing Beast-Mode NSE Scan across liquid universe...")
    angel = Angel()
    signals = []

    today_emitted = get_today_emitted_keys()

    for s in watchlist():
        try:
            m = get_metrics(s, angel)
            sig = classify(m) if m else None
            if sig and m:
                sig.thesis = ai_thesis(sig, m)
                signals.append(sig)
        except Exception as e:
            LOG.debug("Skipping %s: %s", s, e)

    signals.sort(key=lambda x: x.rr, reverse=True)
    max_sigs = int(os.getenv("MAX_SIGNALS", "10"))
    signals = signals[:max_sigs]

    if not force:
        # Strict deduplication: emit ONLY fresh signals not alerted today
        fresh_signals = [s for s in signals if f"{s.symbol}:{s.action}" not in today_emitted]
    else:
        fresh_signals = signals

    if not fresh_signals:
        if not signals:
            LOG.info("NSE Scan: No qualifying setups found.")
            if force:
                telegram("*NSE LIVE SCAN*\nNo high-conviction BUY/SHORT setup currently meets our strict risk rules.")
        else:
            LOG.info("NSE Scan: %d setups detected, all already alerted today (0 duplicate spam).", len(signals))
        return

    message = "🚨 *NSE FRESH LIVE SIGNALS*\n─────────────────────────\n\n" + "\n\n─────────────────────────\n\n".join(card(s) for s in fresh_signals)
    for i in range(0, len(message), 3800):
        telegram(message[i:i+3800])

    log_signals(fresh_signals)
    LOG.info("Dispatched %d fresh NSE signal cards", len(fresh_signals))

def index(ticker: str):
    d = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    c = d.Close.astype(float)
    return float(c.iloc[-1]), float((c.iloc[-1] / c.iloc[-2] - 1) * 100), float(c.tail(20).min()), float(c.tail(20).max())

def briefing():
    try:
        n, b = index("^NSEI"), index("^NSEBANK")
        bias = "BULLISH" if n[1] >= 0 and b[1] >= 0 else "BEARISH" if n[1] < 0 and b[1] < 0 else "NEUTRAL"
        telegram(
            f"☀️ *NSE PRE-MARKET CUES*\n"
            f"NIFTY 50: ₹{n[0]:,.2f} ({n[1]:+.2f}%)\n"
            f"Support / Resistance: ₹{n[2]:,.0f} / ₹{n[3]:,.0f}\n\n"
            f"BANK NIFTY: ₹{b[0]:,.2f} ({b[1]:+.2f}%)\n"
            f"Support / Resistance: ₹{b[2]:,.0f} / ₹{b[3]:,.0f}\n\n"
            f"Opening Bias: *{bias}*\n"
            f"Focus Sectors: Banking, IT, Energy & Auto."
        )
    except Exception as e:
        LOG.error("Briefing failed: %s", e)

def eod():
    """Evaluate and broadcast end-of-day performance scorecard for all signals issued today."""
    today = datetime.now().date().isoformat()
    today_records = []
    if SIGNAL_LOG.exists():
        for line in SIGNAL_LOG.read_text(encoding="utf8").splitlines():
            try:
                rec = json.loads(line)
                if str(rec.get("created_at", "")).startswith(today):
                    today_records.append(rec)
            except Exception:
                pass

    # Deduplicate records by symbol + action
    unique_signals = {}
    for r in today_records:
        key = f"{r.get('symbol')}:{r.get('action')}"
        if key not in unique_signals:
            unique_signals[key] = r
    signals_to_audit = list(unique_signals.values())

    try:
        n = index("^NSEI")
        market_change = n[1]
        market_status = "POSITIVE" if market_change > 0 else "NEGATIVE" if market_change < 0 else "FLAT"
    except Exception:
        n = (22620.0, 0.0, 22000.0, 23000.0)
        market_change = 0.0
        market_status = "FLAT"

    if not signals_to_audit:
        telegram(
            f"🏁 *NSE END-OF-DAY SCORECARD — {today}*\n"
            f"═════════════════════════════════════\n"
            f"📊 NIFTY 50: ₹{n[0]:,.2f} ({market_change:+.2f}%)\n"
            f"Market Session: *{market_status}*\n"
            f"Signals Issued Today: *0*\n"
            f"─────────────────────────────────────\n"
            f"No trades were alerted during today's session."
        )
        return

    wins = 0
    losses = 0
    open_trades = 0
    total_pnl_pct = 0.0
    audit_lines = []

    for s in signals_to_audit:
        sym = s.get("symbol", "")
        action = s.get("action", "BUY")
        style = s.get("style", "INTRADAY")
        entry = (float(s.get("entry_low", 0)) + float(s.get("entry_high", 0))) / 2.0
        t1 = float(s.get("target1", 0))
        t2 = float(s.get("target2", 0))
        stop = float(s.get("stop", 0))

        # Fetch today's price action from yfinance
        try:
            d = yf_history(sym, "5d")
            if d.empty:
                raise ValueError("No price data")
            day_high = float(d["High"].iloc[-1])
            day_low = float(d["Low"].iloc[-1])
            day_close = float(d["Close"].iloc[-1])
        except Exception:
            day_high = day_close = day_low = entry

        is_bullish = ("BUY" in action.upper()) or ("ACCUMULATE" in action.upper())

        if is_bullish:
            if day_high >= t2:
                pnl = ((t2 - entry) / entry) * 100
                outcome = f"🚀 *TARGET 2 HIT* (+{pnl:.1f}%)"
                wins += 1
            elif day_high >= t1:
                pnl = ((t1 - entry) / entry) * 100
                outcome = f"✅ *TARGET 1 HIT* (+{pnl:.1f}%)"
                wins += 1
            elif day_low <= stop:
                pnl = -abs(((entry - stop) / entry) * 100)
                outcome = f"🛑 *STOP LOSS HIT* ({pnl:.1f}%)"
                losses += 1
            else:
                pnl = ((day_close - entry) / entry) * 100
                outcome = f"⏳ In Profit ({pnl:+.1f}%)" if pnl >= 0 else f"⏳ In Drawdown ({pnl:+.1f}%)"
                open_trades += 1
        else:  # SHORT
            if day_low <= t2:
                pnl = ((entry - t2) / entry) * 100
                outcome = f"🚀 *TARGET 2 HIT* (+{pnl:.1f}%)"
                wins += 1
            elif day_low <= t1:
                pnl = ((entry - t1) / entry) * 100
                outcome = f"✅ *TARGET 1 HIT* (+{pnl:.1f}%)"
                wins += 1
            elif day_high >= stop:
                pnl = -abs(((stop - entry) / entry) * 100)
                outcome = f"🛑 *STOP LOSS HIT* ({pnl:.1f}%)"
                losses += 1
            else:
                pnl = ((entry - day_close) / entry) * 100
                outcome = f"⏳ In Profit ({pnl:+.1f}%)" if pnl >= 0 else f"⏳ In Drawdown ({pnl:+.1f}%)"
                open_trades += 1

        total_pnl_pct += pnl
        audit_lines.append(
            f"• *{sym}* ({style} {action})\n"
            f"  Entry: ₹{entry:,.1f} | High: ₹{day_high:,.1f} | Close: ₹{day_close:,.1f}\n"
            f"  Result: {outcome}"
        )

    completed = wins + losses
    win_rate = (wins / completed * 100) if completed > 0 else 100.0 if wins > 0 else 0.0

    summary_text = (
        f"🏁 *NSE END-OF-DAY TRADE SCORECARD — {today}*\n"
        f"═════════════════════════════════════\n"
        f"📊 NIFTY 50: ₹{n[0]:,.2f} ({market_change:+.2f}%)\n"
        f"Session: *{market_status}*\n"
        f"─────────────────────────────────────\n"
        f"🎯 *PERFORMANCE OVERVIEW:*\n"
        f"• Total Signals Issued: *{len(signals_to_audit)}*\n"
        f"• ✅ Targets Hit: *{wins}*\n"
        f"• 🛑 Stopped Out: *{losses}*\n"
        f"• ⏳ Open / Carrying: *{open_trades}*\n"
        f"• 🏆 Win Rate: *{win_rate:.1f}%* ({wins}/{completed} completed)\n"
        f"• 💰 Net Signal Performance: *{total_pnl_pct:+.2f}%*\n"
        f"─────────────────────────────────────\n"
        f"📋 *TRADE BY TRADE AUDIT:*\n"
        + "\n\n".join(audit_lines) +
        f"\n═════════════════════════════════════\n"
        f"⚠️ _Paper observation tracking. Review trailing levels for carried positions._"
    )

    for i in range(0, len(summary_text), 3800):
        telegram(summary_text[i:i+3800])

    LOG.info(
        "EOD Scorecard dispatched: %d signals, Win rate: %.1f%%, Total P&L: %+.2f%%",
        len(signals_to_audit), win_rate, total_pnl_pct
    )

def main():
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "scan"
    force_flag = "--force" in sys.argv or "-f" in sys.argv
    if command == "scan":
        scan(force=force_flag)
    elif command == "briefing":
        briefing()
    elif command == "eod":
        eod()
    elif command in ("dashboard", "web", "server"):
        import web_server
        web_server.main()
    else:
        raise SystemExit("Usage: python cli.py [scan|briefing|eod|dashboard] [--force]")

if __name__ == "__main__":
    main()

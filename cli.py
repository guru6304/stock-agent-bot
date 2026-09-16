import io
import os
import sys
import pandas as pd
import pytz
import requests
import yfinance as yf
from dotenv import load_dotenv
from google import genai

# Load local .env if available
load_dotenv()

# --- CONFIGURATION (DYNAMIC WITH YOUR PROVIDED DEFAULTS) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8604600704:AAFF9plPcXdpNv9pGOmscrqTQpL9oxgsu7E")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "893713256")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "daknc1pr01qln1kf4t00daknc1pr01qln1kf4t0g")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "2f8dda7420574a89b673c2a1ec692548")

TIMEZONE_STR = os.getenv("TIMEZONE", "Asia/Kolkata")
IST = pytz.timezone(TIMEZONE_STR)
MARKET_OPEN_HOUR = int(os.getenv("MARKET_OPEN_HOUR", 9))
MARKET_OPEN_MINUTE = int(os.getenv("MARKET_OPEN_MINUTE", 15))
MARKET_CLOSE_HOUR = int(os.getenv("MARKET_CLOSE_HOUR", 15))
MARKET_CLOSE_MINUTE = int(os.getenv("MARKET_CLOSE_MINUTE", 30))
RUN_INTERVAL_MINUTES = int(os.getenv("RUN_INTERVAL_MINUTES", 15))

USE_SQLITE = int(os.getenv("USE_SQLITE", 1))
DB_PATH = os.getenv("DB_PATH", "logs/stock_agent.db")
PAPER_STARTING_CAPITAL = float(os.getenv("PAPER_STARTING_CAPITAL", 100000))
PAPER_MAX_POSITIONS = int(os.getenv("PAPER_MAX_POSITIONS", 10))
SIGNAL_THRESHOLD = int(os.getenv("SIGNAL_THRESHOLD", 4))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AQ.Ab8RN6JeTLaJzF_iA2kFsSu5PFYU7r3aFmRrVbzjZ7XdWAcAPg")


def send_telegram(text: str):
    """Dispatches markdown trade alerts directly to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Missing Telegram credentials.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        res = requests.post(url, json=payload, timeout=12)
        print(f"Telegram status code: {res.status_code}")
    except Exception as e:
        print(f"Failed to push Telegram alert: {e}")


def ask_gemini(prompt: str) -> str:
    """Invokes Gemini 2.5 Flash for trade setup synthesis."""
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured."
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return "AI analysis could not be generated."


def get_dynamic_watchlist():
    """
    Dynamically loads the stock list:
    1. Checks if a custom comma-separated WATCHLIST environment variable exists.
    2. Otherwise, fetches the live official NIFTY 50 constituent CSV directly from NSE India archives.
    """
    env_watchlist = os.getenv("WATCHLIST")
    if env_watchlist:
        return [s.strip() for s in env_watchlist.split(",") if s.strip()]

    print("Fetching live constituent list dynamically from NSE India...")
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text))
            symbols = [f"{sym}.NS" for sym in df["Symbol"].dropna().tolist()]
            print(f"Loaded {len(symbols)} dynamic NSE constituents.")
            return symbols
    except Exception as e:
        print(f"Warning: Could not fetch live NSE CSV ({e}). Using liquid index basket.")

    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "TATAMOTORS.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS"
    ]


def indian_briefing():
    """Dynamically calculates live NIFTY & BANK NIFTY performance and generates the morning brief."""
    print("Fetching Indian index session cues...")
    nifty = yf.Ticker("^NSEI").history(period="5d")
    banknifty = yf.Ticker("^NSEBANK").history(period="5d")

    n_close = nifty["Close"].iloc[-1] if not nifty.empty else 0.0
    n_prev = nifty["Close"].iloc[-2] if len(nifty) > 1 else n_close
    n_chg = round(((n_close - n_prev) / n_prev) * 100, 2) if n_prev else 0.0

    bn_close = banknifty["Close"].iloc[-1] if not banknifty.empty else 0.0
    bn_prev = banknifty["Close"].iloc[-2] if len(banknifty) > 1 else bn_close
    bn_chg = round(((bn_close - bn_prev) / bn_prev) * 100, 2) if bn_prev else 0.0

    prompt = f"""
    You are an Indian Equity Technical Strategist.
    Analyze these live pre-market index cues for the National Stock Exchange (NSE):
    - NIFTY 50: {n_close:.2f} ({n_chg:+0.2f}%)
    - BANK NIFTY: {bn_close:.2f} ({bn_chg:+0.2f}%)
    - Virtual Trading Capital: ₹{PAPER_STARTING_CAPITAL:,.0f}

    Provide a concise Telegram pre-market briefing card:
    📊 *INDIAN MARKET PRE-MARKET BRIEFING*
    • NIFTY 50 Key Support & Resistance Levels
    • BANK NIFTY Key Support & Resistance Levels
    • Expected Opening Bias (Bullish / Bearish / Neutral)
    • 3 Key Indian Sectors to Watch Today
    """
    card = ask_gemini(prompt)
    send_telegram(card)
    print("Pre-market briefing delivered to Telegram.")


def indian_scan():
    """Scans the dynamic stock list for momentum and volume expansion."""
    watchlist = get_dynamic_watchlist()
    print(f"Scanning {len(watchlist)} Indian equities for setups...")
    alerts = []

    for ticker in watchlist:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1mo")
            if len(df) < 20:
                continue

            close = df["Close"].iloc[-1]
            sma20 = df["Close"].tail(20).mean()
            vol = df["Volume"].iloc[-1]
            avg_vol = df["Volume"].tail(20).mean()

            # Technical breakout check: Price > 20 SMA with volume expansion
            if close > sma20 and vol > (avg_vol * 1.15):
                name = ticker.replace(".NS", "")
                alerts.append(f"• *{name}*: ₹{close:.2f} (Above 20-SMA with 1.15x Volume Surge)")
        except Exception:
            continue

    if not alerts:
        send_telegram("🇮🇳 *NSE Market Scan*: No volume breakout setups triggered right now.")
        return

    top_alerts = alerts[:PAPER_MAX_POSITIONS]
    stocks_text = "\n".join(top_alerts)
    prompt = f"""
    You are an Indian Equity Technical Analyst.
    Breakout stocks detected on the NSE:
    {stocks_text}

    Generate an actionable Telegram recommendation card for Indian swing/intraday traders:
    - Primary setup pick
    - Suggested Entry Range, Stop-Loss (in ₹), Target 1 & Target 2
    - 1-sentence technical trade rationale
    """
    recommendation = ask_gemini(prompt)
    send_telegram(f"🇮🇳 *NSE BREAKOUT RADAR*\n\n{stocks_text}\n\n{recommendation}")
    print("Scan results delivered to Telegram.")


def indian_eod():
    """Generates dynamic End-of-Day report for the NSE session."""
    print("Fetching End-of-Day summary for NSE...")
    nifty = yf.Ticker("^NSEI").history(period="5d")
    if not nifty.empty:
        close = nifty["Close"].iloc[-1]
        prev = nifty["Close"].iloc[-2] if len(nifty) > 1 else close
        chg = round(((close - prev) / prev) * 100, 2) if prev else 0.0
        summary = (
            f"🏁 *NSE END-OF-DAY SUMMARY*\n\n"
            f"• *NIFTY 50*: ₹{close:.2f} ({chg:+0.2f}%)\n"
            f"• Active Capital Pool: ₹{PAPER_STARTING_CAPITAL:,.0f}\n"
            f"• Market session closed at {MARKET_CLOSE_HOUR}:{MARKET_CLOSE_MINUTE} IST. "
            f"Scanners will resume at {MARKET_OPEN_HOUR}:{MARKET_OPEN_MINUTE} AM IST tomorrow."
        )
        send_telegram(summary)
    print("EOD summary delivered to Telegram.")


if __name__ == "__main__":
    args = [a.lower() for a in sys.argv[1:] if not a.startswith("--")]
    cmd = args[0] if args else "scan"

    if cmd == "briefing":
        indian_briefing()
    elif cmd == "scan":
        indian_scan()
    elif cmd == "eod":
        indian_eod()
    else:
        indian_scan()
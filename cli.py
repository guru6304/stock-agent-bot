"""Real-time NSE Stock Recommendation Engine: scan, briefing, eod."""
from __future__ import annotations
import csv, io, json, logging, os, re, sys
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

load_dotenv(); logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
LOG=logging.getLogger("nse-engine"); HTTP=requests.Session()
HTTP.headers.update({"User-Agent":"Mozilla/5.0 (NSE Stock Recommendation Engine)"})
NSE_CSV="https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
SCRIP_MASTER="https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
SIGNAL_LOG=Path(os.getenv("SIGNAL_LOG","logs/nse_signals.jsonl"))
def env(k:str)->str:return os.getenv(k,"").strip()
def symbol(x:str)->str:return x.upper().strip().replace(".NS","").replace("-EQ","")

@dataclass
class Metrics:
    symbol:str; price:float; high:float; low:float; sma20:float; ema20:float; ema50:float; ema200:float; vol_ratio:float; high3:float; high20:float; atr:float; vwap:Optional[float]; orb:Optional[float]; source:str; quality:bool
@dataclass
class Signal:
    symbol:str; style:str; action:str; entry_low:float; entry_high:float; stop:float; target1:float; target2:float; rr:float; thesis:str; source:str

def watchlist()->list[str]:
    if env("WATCHLIST"): return list(dict.fromkeys(symbol(x) for x in env("WATCHLIST").split(",") if symbol(x)))
    try:
        r=HTTP.get(NSE_CSV,timeout=15); r.raise_for_status()
        out=[symbol(row.get("Symbol", "")) for row in csv.DictReader(io.StringIO(r.text))]
        if any(out): return [x for x in out if x]
    except (requests.RequestException,csv.Error) as e: LOG.warning("NSE constituent list unavailable: %s",e)
    # Dynamic discovery from Scrip Master instead of hardcoded symbols
    try:
        from instrument_discovery import DynamicInstrumentDiscovery
        disc = DynamicInstrumentDiscovery()
        return [inst.symbol for inst in disc.get_equity_universe(limit=50)]
    except Exception as e:
        LOG.error("Dynamic discovery failed: %s", e)
        return []

class Angel:
    """SmartAPI authentication and LTP source; failures deliberately fall back."""
    def __init__(self):
        self.api=None; self.tokens={}
        required=("ANGEL_API_KEY","ANGEL_CLIENT_CODE","ANGEL_PIN","ANGEL_TOTP_KEY")
        if not all(env(x) for x in required): return
        if not SmartConnect or not pyotp:
            LOG.warning("Angel credentials found but smartapi-python/pyotp are not installed"); return
        try:
            self.api=SmartConnect(api_key=env("ANGEL_API_KEY"))
            otp=pyotp.TOTP(env("ANGEL_TOTP_KEY")).now()
            session=self.api.generateSession(env("ANGEL_CLIENT_CODE"),env("ANGEL_PIN"),otp)
            if not session or not session.get("status"): raise RuntimeError((session or {}).get("message","login rejected"))
            data=HTTP.get(SCRIP_MASTER,timeout=25).json()
            self.tokens={x["symbol"].replace("-EQ",""):str(x["token"]) for x in data if x.get("exch_seg")=="NSE" and str(x.get("symbol","")).endswith("-EQ")}
            LOG.info("Angel One session authenticated")
        except Exception as e: self.api=None; LOG.warning("Angel unavailable; falling back to yfinance NSE: %s",e)
    def quote(self,s:str)->Optional[float]:
        if not self.api or s not in self.tokens:return None
        try:
            q=self.api.ltpData("NSE",s+"-EQ",self.tokens[s]); value=(q or {}).get("data",{}).get("ltp")
            return float(value) if value else None
        except Exception as e: LOG.info("Angel quote limit/error for %s: %s",s,e); return None

def yf_history(s:str,period:str,interval="1d")->pd.DataFrame:
    try:
        d=yf.download(s+".NS",period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
        if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
        return d.dropna(how="all")
    except Exception as e: LOG.info("NSE fallback failed %s: %s",s,e); return pd.DataFrame()

def get_metrics(s:str,angel:Angel)->Optional[Metrics]:
    d=yf_history(s,"2y")
    if len(d)<205 or not {"Close","High","Low","Volume"}.issubset(d): return None
    c,h,l,v=(d[x].astype(float) for x in ("Close","High","Low","Volume")); live=angel.quote(s); price=live or float(c.iloc[-1])
    e20,e50,e200=(float(c.ewm(span=n,adjust=False).mean().iloc[-1]) for n in (20,50,200))
    tr=pd.concat((h-l,(h-c.shift()).abs(),(l-c.shift()).abs()),axis=1).max(axis=1)
    intra=yf_history(s,"5d","5m"); vw=orb=None
    if not intra.empty and {"High","Low","Close","Volume"}.issubset(intra):
        typical=(intra.High+intra.Low+intra.Close)/3; denom=intra.Volume.sum()
        vw=float((typical*intra.Volume).sum()/denom) if denom else None; orb=float(intra.High.tail(15).max())
    quality=price>e200 and float(c.pct_change(252).iloc[-1])>-0.20
    return Metrics(s,price,float(h.iloc[-1]),float(l.iloc[-1]),float(c.tail(20).mean()),e20,e50,e200,float(v.iloc[-1]/max(v.tail(21).iloc[:-1].mean(),1)),float(h.tail(4).iloc[:-1].max()),float(h.tail(21).iloc[:-1].max()),float(tr.tail(14).mean()),vw,orb,"Angel One SmartAPI" if live else "yfinance NSE",quality)

def classify(m:Metrics)->Optional[Signal]:
    p,atr=m.price,max(m.atr,p*.003); style=action=thesis=None
    if m.vwap and m.orb and p>m.vwap and p>=m.orb*.998 and m.vol_ratio>=1.5:
        style,action,stop,thesis="INTRADAY","BUY",min(m.vwap,p*.987),"VWAP reclaim is holding as price challenges the opening-range high on expanded participation."
    elif p>m.high3 and p>m.ema20 and m.vol_ratio>=1.5:
        style,action,stop,thesis="SHORT-TERM","BUY",min(m.ema20,p-1.2*atr),"A multi-day high breakout is confirmed by at least 1.5x normal daily volume."
    elif p>m.high20 and m.ema20>m.ema50 and m.vol_ratio>=1.15:
        style,action,stop,thesis="SWING","BUY",min(m.ema20,p-1.5*atr),"A consolidation breakout aligns with the rising 20/50-EMA trend."
    elif m.quality and abs(p/m.ema200-1)<=.035 and m.ema20>m.ema50:
        style,action,stop,thesis="LONG-TERM","ACCUMULATE",p-2.5*atr,"Price is retesting rising 200-day EMA support with a positive long-term trend anchor."
    else:return None
    if style=="INTRADAY":stop=min(stop,p*.992)
    risk=p-stop
    if risk<=0:return None
    t1,t2=p+2*risk,p+3*risk
    if style=="LONG-TERM":t1,t2=max(t1,p*1.15),max(t2,p*1.25)
    return Signal(m.symbol,style,action,p*.997,p*1.003,stop,t1,t2,(t1-p)/risk,thesis,m.source)

def ai_thesis(s:Signal,m:Metrics)->str:
    """Gemini synthesizes the card trigger from every required technical input."""
    if not env("GEMINI_API_KEY") or not genai:return s.thesis
    prompt=("You are the synthesis step for a concrete NSE trade card. Return exactly one factual "
      "technical-trigger sentence, no greeting, disclaimer, or prediction. The card's fixed values are: "
      f"Ticker={m.symbol}; Trading Style={s.style}; Action={s.action}; Entry=₹{s.entry_low:.2f}-₹{s.entry_high:.2f}; "
      f"Stop=₹{s.stop:.2f}; Target1=₹{s.target1:.2f}; Target2=₹{s.target2:.2f}; R:R=1:{s.rr:.1f}. "
      f"Indicators: Price={m.price:.2f}; SMA20={m.sma20:.2f}; EMA20={m.ema20:.2f}; EMA50={m.ema50:.2f}; "
      f"EMA200={m.ema200:.2f}; Volume/20d={m.vol_ratio:.2f}; DayRange={m.low:.2f}-{m.high:.2f}. Base trigger={s.thesis}")
    try:
        text=re.sub(r"\s+"," ",genai.Client(api_key=env("GEMINI_API_KEY")).models.generate_content(model="gemini-2.5-flash",contents=prompt).text.strip())
        return text if text and len(text)<=240 else s.thesis
    except Exception as e:LOG.warning("Gemini failed for %s: %s",m.symbol,e);return s.thesis

def card(s:Signal)->str:
    return (f"*{s.symbol} — {s.style}*\nAction: *{s.action}*\nEntry: ₹{s.entry_low:,.2f}–₹{s.entry_high:,.2f}\nStrict Stop-Loss: ₹{s.stop:,.2f}\nTarget 1: ₹{s.target1:,.2f} | Target 2: ₹{s.target2:,.2f}\nRisk:Reward: *1:{s.rr:.1f}*\nTrigger: {s.thesis}\nData: {s.source}")
def telegram(text:str)->bool:
    if not env("TELEGRAM_BOT_TOKEN") or not env("TELEGRAM_CHAT_ID"):LOG.error("Telegram credentials missing");return False
    try:
        r=HTTP.post(f"https://api.telegram.org/bot{env('TELEGRAM_BOT_TOKEN')}/sendMessage",json={"chat_id":env("TELEGRAM_CHAT_ID"),"text":text,"parse_mode":"Markdown","disable_web_page_preview":True},timeout=15);r.raise_for_status();return True
    except requests.RequestException as e:LOG.error("Telegram dispatch failed: %s",e);return False
def log_signals(ss:list[Signal]):
    SIGNAL_LOG.parent.mkdir(parents=True,exist_ok=True)
    with SIGNAL_LOG.open("a",encoding="utf8") as f:
        for s in ss:f.write(json.dumps({"created_at":datetime.now().astimezone().isoformat(),**asdict(s)})+"\n")

def scan():
    angel=Angel(); signals=[]
    for s in watchlist():
        try:
            m=get_metrics(s,angel); signal=classify(m) if m else None
            if signal and m:signal.thesis=ai_thesis(signal,m);signals.append(signal)
        except Exception as e:LOG.warning("Skipping %s: %s",s,e)
    signals.sort(key=lambda x:x.rr,reverse=True); signals=signals[:int(os.getenv("MAX_SIGNALS","8"))]
    if not signals: telegram("*NSE LIVE SCAN*\nNo high-conviction BUY/ACCUMULATE setup meets the defined risk and confirmation rules.");return
    message="*NSE LIVE TRADE SIGNALS*\n\n"+"\n\n".join(card(s) for s in signals)
    for i in range(0,len(message),3800):telegram(message[i:i+3800])
    log_signals(signals);LOG.info("Dispatched %d NSE cards",len(signals))

def index(ticker:str):
    d=yf.download(ticker,period="6mo",auto_adjust=True,progress=False)
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    c=d.Close.astype(float);return float(c.iloc[-1]),float((c.iloc[-1]/c.iloc[-2]-1)*100),float(c.tail(20).min()),float(c.tail(20).max())
def briefing():
    try:
        n,b=index("^NSEI"),index("^NSEBANK"); bias="BULLISH" if n[1]>=0 and b[1]>=0 else "BEARISH" if n[1]<0 and b[1]<0 else "NEUTRAL"
        telegram(f"*NSE PRE-MARKET CUES*\nNIFTY 50: ₹{n[0]:,.2f} ({n[1]:+.2f}%)\nSupport / Resistance: ₹{n[2]:,.0f} / ₹{n[3]:,.0f}\n\nBANK NIFTY: ₹{b[0]:,.2f} ({b[1]:+.2f}%)\nSupport / Resistance: ₹{b[2]:,.0f} / ₹{b[3]:,.0f}\n\nOpening bias: *{bias}*\nFocus sectors: Banks, IT, Energy.")
    except Exception as e:LOG.error("Briefing failed: %s",e)
def eod():
    records=[]
    if SIGNAL_LOG.exists():
        for line in SIGNAL_LOG.read_text(encoding="utf8").splitlines():
            try:records.append(json.loads(line))
            except json.JSONDecodeError:pass
    try:
        n=index("^NSEI"); today=datetime.now().date().isoformat(); count=sum(str(x.get("created_at","")).startswith(today) for x in records); status="POSITIVE" if n[1]>0 else "NEGATIVE" if n[1]<0 else "FLAT"
        telegram(f"*NSE END-OF-DAY STATUS*\nNIFTY 50 close: ₹{n[0]:,.2f} ({n[1]:+.2f}%)\nSession: *{status}*\nSignals issued today: *{count}*\nClosed setups: *0* (broker execution is not assumed)\n\nReview stop-losses and trailing levels before the next session.")
    except Exception as e:LOG.error("EOD failed: %s",e)
def main():
    command=sys.argv[1].lower() if len(sys.argv)>1 else "scan"
    if command=="scan":scan()
    elif command=="briefing":briefing()
    elif command in ("dashboard", "web", "server"):
        import web_server
        web_server.main()
    else:raise SystemExit("Usage: python cli.py [scan|briefing|eod|dashboard]")
if __name__=="__main__":main()

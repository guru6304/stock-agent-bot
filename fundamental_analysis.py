"""
Fundamental Analysis Module — scores tickers on 15 institutional-grade criteria.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except ImportError:
    SentimentIntensityAnalyzer = None
    vader = None
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from data_layer import fetch_fundamentals, fetch_news

# Sector median P/E ratios (approximate, used as baseline)
SECTOR_MEDIAN_PE = {
    "Technology": 30,
    "Healthcare": 22,
    "Financials": 14,
    "Consumer Discretionary": 25,
    "Consumer Staples": 20,
    "Energy": 12,
    "Industrials": 20,
    "Materials": 16,
    "Utilities": 18,
    "Real Estate": 35,
    "Communication Services": 18,
}

@dataclass
class FundamentalSignals:
    """Container for fundamental analysis results."""
    # --- Existing 6 criteria ---
    ticker: str = ""
    pe_ratio: Optional[float] = None
    pe_below_sector_median: bool = False
    eps_growth_yoy: Optional[float] = None
    eps_growth_above_10: bool = False
    revenue_growth_qoq: Optional[float] = None
    revenue_growth_above_5: bool = False
    debt_to_equity: Optional[float] = None
    debt_to_equity_healthy: bool = False
    analyst_consensus: Optional[str] = None
    analyst_buy_or_better: bool = False
    news_sentiment_score: float = 0.0
    news_sentiment_positive: bool = False

    # --- Valuation ---
    forward_pe: Optional[float] = None
    forward_pe_attractive: bool = False
    peg_ratio: Optional[float] = None
    peg_ratio_attractive: bool = False
    price_to_sales: Optional[float] = None
    price_to_book: Optional[float] = None
    ev_to_ebitda: Optional[float] = None

    # --- Profitability ---
    roe: Optional[float] = None
    roe_strong: bool = False
    roic: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    margin_expanding: bool = False
    fcf_yield: Optional[float] = None
    fcf_yield_strong: bool = False

    # --- Balance Sheet ---
    current_ratio: Optional[float] = None
    current_ratio_healthy: bool = False
    interest_coverage: Optional[float] = None

    # --- Market Dynamics ---
    short_interest_pct: Optional[float] = None
    short_squeeze_risk: bool = False
    insider_buy_count: int = 0
    insider_sell_count: int = 0
    insider_net_bullish: bool = False
    institutional_ownership_pct: Optional[float] = None

    # --- Earnings Quality ---
    earnings_surprise_avg: Optional[float] = None
    earnings_beat_streak: int = 0
    earnings_quality_strong: bool = False

    # --- Dividend ---
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

    # --- Composite ---
    fundamental_score: int = 0
    score_details: list = field(default_factory=list)


def _fetch_extended_fundamentals(ticker: str) -> dict:
    """Fetch extended fundamental data directly from yfinance."""
    try:
        import yfinance as yf
        from data_layer import resolve_ticker
        resolved = resolve_ticker(ticker)
        tk = yf.Ticker(resolved)
        info = None
        try:
            info = tk.info
        except Exception:
            pass
        if not info and resolved != ticker:
            try:
                tk = yf.Ticker(ticker)
                info = tk.info
            except Exception:
                pass
        info = info or {}

        if not info:
            return {
                "forward_pe": None, "peg_ratio": None, "price_to_sales": None,
                "price_to_book": None, "ev_to_ebitda": None, "roe": None,
                "roic": None, "gross_margin": None, "operating_margin": None,
                "fcf": None, "market_cap": None, "current_ratio": None,
                "short_pct": None, "institutional_pct": None, "dividend_yield": None,
                "payout_ratio": None, "trailing_pe": None, "interest_coverage": None,
                "earnings_surprises": [], "insider_buys": 0, "insider_sells": 0,
            }

        result = {
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "price_to_book": info.get("priceToBook"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "roe": info.get("returnOnEquity"),
            "roic": info.get("returnOnAssets"),  # proxy for ROIC
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "fcf": info.get("freeCashflow"),
            "market_cap": info.get("marketCap"),
            "current_ratio": info.get("currentRatio"),
            "short_pct": info.get("shortPercentOfFloat"),
            "institutional_pct": info.get("heldPercentInstitutions"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "trailing_pe": info.get("trailingPE"),
            "interest_coverage": info.get("interestCoverage"),
        }

        # Earnings surprise
        try:
            eh = tk.earnings_history
            if eh is not None and not eh.empty:
                surprises = eh["surprisePercent"].dropna().tolist()
                result["earnings_surprises"] = surprises[-4:]  # last 4 quarters
            else:
                result["earnings_surprises"] = []
        except Exception:
            result["earnings_surprises"] = []

        # Insider transactions
        try:
            ins = tk.insider_transactions
            if ins is not None and not ins.empty:
                from datetime import datetime, timedelta
                cutoff = datetime.now() - timedelta(days=90)
                recent = ins[ins.index >= cutoff] if hasattr(ins.index, 'tz') else ins
                buys = 0
                sells = 0
                if "Transaction" in recent.columns:
                    buys = len(recent[recent["Transaction"].str.contains("Purchase|Buy", case=False, na=False)])
                    sells = len(recent[recent["Transaction"].str.contains("Sale|Sell", case=False, na=False)])
                result["insider_buys"] = buys
                result["insider_sells"] = sells
            else:
                result["insider_buys"] = 0
                result["insider_sells"] = 0
        except Exception:
            result["insider_buys"] = 0
            result["insider_sells"] = 0

        return result
    except Exception as e:
        logger.debug("Extended fundamentals fetch failed for %s: %s", ticker, e)
        return {}


def score_news_sentiment(articles: List[Dict]) -> float:
    """Score news sentiment using both VADER and TextBlob, averaged.

    Returns a score between -1 and +1.
    """
    if not articles:
        return 0.0

    scores = []
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        if not text.strip():
            continue

        item_scores = []
        # VADER
        if vader:
            item_scores.append(vader.polarity_scores(text)["compound"])
        # TextBlob
        if TextBlob:
            item_scores.append(TextBlob(text).sentiment.polarity)

        if item_scores:
            scores.append(sum(item_scores) / len(item_scores))

    return sum(scores) / len(scores) if scores else 0.0


def analyze(ticker: str, sector: str = "Technology") -> FundamentalSignals:
    """Run fundamental analysis on a single ticker.

    Returns a FundamentalSignals object with a score from 0-15.
    """
    signals = FundamentalSignals(ticker=ticker)
    details = []

    # Fetch fundamental data
    fundamentals = fetch_fundamentals(ticker)

    # Fetch extended data from yfinance
    ext = _fetch_extended_fundamentals(ticker)

    # =========================================================================
    # EXISTING 6 CRITERIA (1-6)
    # =========================================================================

    # 1. P/E ratio vs sector median
    signals.pe_ratio = fundamentals.get("pe_ratio")
    median_pe = SECTOR_MEDIAN_PE.get(sector, 20)
    if signals.pe_ratio is not None and signals.pe_ratio > 0:
        signals.pe_below_sector_median = signals.pe_ratio < median_pe
        if signals.pe_below_sector_median:
            details.append(f"P/E {signals.pe_ratio:.1f} below sector median {median_pe}")

    # 2. EPS growth YoY > 10%
    signals.eps_growth_yoy = fundamentals.get("eps_growth_yoy")
    if signals.eps_growth_yoy is not None:
        signals.eps_growth_above_10 = signals.eps_growth_yoy > 10
        if signals.eps_growth_above_10:
            details.append(f"EPS growth YoY: {signals.eps_growth_yoy:.1f}%")

    # 3. Revenue growth QoQ > 5%
    signals.revenue_growth_qoq = fundamentals.get("revenue_growth_qoq")
    if signals.revenue_growth_qoq is not None:
        signals.revenue_growth_above_5 = signals.revenue_growth_qoq > 5
        if signals.revenue_growth_above_5:
            details.append(f"Revenue growth QoQ: {signals.revenue_growth_qoq:.1f}%")

    # 4. Debt-to-equity < 1.0
    signals.debt_to_equity = fundamentals.get("debt_to_equity")
    if signals.debt_to_equity is not None:
        signals.debt_to_equity_healthy = signals.debt_to_equity < 1.0
        if signals.debt_to_equity_healthy:
            details.append(f"D/E ratio: {signals.debt_to_equity:.2f}")

    # 5. Analyst consensus = Buy or Strong Buy
    raw_consensus = fundamentals.get("analyst_consensus", "")
    if isinstance(raw_consensus, str):
        signals.analyst_consensus = raw_consensus.lower()
        signals.analyst_buy_or_better = signals.analyst_consensus in (
            "buy", "strong_buy", "strongbuy", "strong buy",
        )
    if signals.analyst_buy_or_better:
        details.append(f"Analyst consensus: {signals.analyst_consensus}")

    # 6. News sentiment > 0.2
    articles = fetch_news(ticker)
    signals.news_sentiment_score = score_news_sentiment(articles)
    signals.news_sentiment_positive = signals.news_sentiment_score > 0.2
    if signals.news_sentiment_positive:
        details.append(f"News sentiment: {signals.news_sentiment_score:.2f}")

    # =========================================================================
    # NEW 9 CRITERIA (7-15) — Institutional-Grade
    # =========================================================================

    # Populate extended fields on signals (informational, even if not scored)
    signals.forward_pe = ext.get("forward_pe")
    signals.peg_ratio = ext.get("peg_ratio")
    signals.price_to_sales = ext.get("price_to_sales")
    signals.price_to_book = ext.get("price_to_book")
    signals.ev_to_ebitda = ext.get("ev_to_ebitda")
    signals.roe = ext.get("roe")
    signals.roic = ext.get("roic")
    signals.gross_margin = ext.get("gross_margin")
    signals.operating_margin = ext.get("operating_margin")
    signals.current_ratio = ext.get("current_ratio")
    signals.short_interest_pct = ext.get("short_pct")
    signals.institutional_ownership_pct = ext.get("institutional_pct")
    signals.dividend_yield = ext.get("dividend_yield")
    signals.payout_ratio = ext.get("payout_ratio")
    signals.interest_coverage = ext.get("interest_coverage")
    signals.insider_buy_count = ext.get("insider_buys", 0)
    signals.insider_sell_count = ext.get("insider_sells", 0)

    # 7. FCF Yield > 4%
    try:
        fcf = ext.get("fcf", 0) or 0
        market_cap = ext.get("market_cap", 0) or 0
        if market_cap > 0:
            signals.fcf_yield = fcf / market_cap * 100
            signals.fcf_yield_strong = signals.fcf_yield > 4.0
            if signals.fcf_yield_strong:
                details.append(f"FCF yield: {signals.fcf_yield:.1f}%")
    except Exception:
        pass

    # 8. PEG Ratio < 1.5
    try:
        peg = ext.get("peg_ratio")
        if peg is not None and peg > 0:
            signals.peg_ratio_attractive = peg < 1.5
            if signals.peg_ratio_attractive:
                details.append(f"PEG ratio: {peg:.2f} (undervalued rel. to growth)")
    except Exception:
        pass

    # 9. ROE > 15%
    try:
        roe = ext.get("roe")
        if roe is not None:
            signals.roe_strong = roe > 0.15
            if signals.roe_strong:
                details.append(f"ROE: {roe * 100:.1f}%")
    except Exception:
        pass

    # 10. Margin Expanding (operating margin > 15% as proxy)
    try:
        op_margin = ext.get("operating_margin")
        if op_margin is not None:
            signals.margin_expanding = op_margin > 0.15
            if signals.margin_expanding:
                details.append(f"Operating margin: {op_margin * 100:.1f}% (strong)")
    except Exception:
        pass

    # 11. Current Ratio > 1.5
    try:
        cr = ext.get("current_ratio")
        if cr is not None:
            signals.current_ratio_healthy = cr > 1.5
            if signals.current_ratio_healthy:
                details.append(f"Current ratio: {cr:.2f}")
    except Exception:
        pass

    # 12. Earnings Beat Streak >= 3
    try:
        surprises = ext.get("earnings_surprises", [])
        if surprises:
            signals.earnings_surprise_avg = sum(surprises) / len(surprises)
            streak = 0
            for s in reversed(surprises):
                if s > 0:
                    streak += 1
                else:
                    break
            signals.earnings_beat_streak = streak
            signals.earnings_quality_strong = streak >= 3
            if signals.earnings_quality_strong:
                details.append(f"Earnings beat streak: {streak} quarters")
    except Exception:
        pass

    # 13. Insider Net Buying
    try:
        buys = ext.get("insider_buys", 0)
        sells = ext.get("insider_sells", 0)
        signals.insider_net_bullish = buys > sells and buys > 0
        if signals.insider_net_bullish:
            details.append(f"Insider net buying: {buys} buys vs {sells} sells (90d)")
    except Exception:
        pass

    # 14. Short Interest < 5% (healthy, no major bearish bet)
    short_healthy = False
    try:
        short_pct = ext.get("short_pct")
        if short_pct is not None:
            signals.short_squeeze_risk = short_pct > 0.15
            short_healthy = short_pct < 0.05
            if short_healthy:
                details.append(f"Low short interest: {short_pct * 100:.1f}%")
    except Exception:
        pass

    # 15. Forward P/E < Trailing P/E (growth expected)
    try:
        fwd = ext.get("forward_pe")
        trail = ext.get("trailing_pe")
        if fwd is not None and trail is not None and trail > 0 and fwd > 0:
            signals.forward_pe_attractive = fwd < trail
            if signals.forward_pe_attractive:
                details.append(f"Forward P/E {fwd:.1f} < Trailing P/E {trail:.1f}")
    except Exception:
        pass

    # =========================================================================
    # Composite score (0-15)
    # =========================================================================
    score = sum([
        # Original 6
        signals.pe_below_sector_median,
        signals.eps_growth_above_10,
        signals.revenue_growth_above_5,
        signals.debt_to_equity_healthy,
        signals.analyst_buy_or_better,
        signals.news_sentiment_positive,
        # New 9
        signals.fcf_yield_strong,              # 7
        signals.peg_ratio_attractive,          # 8
        signals.roe_strong,                    # 9
        signals.margin_expanding,              # 10
        signals.current_ratio_healthy,         # 11
        signals.earnings_quality_strong,       # 12
        signals.insider_net_bullish,           # 13
        short_healthy,                         # 14
        signals.forward_pe_attractive,         # 15
    ])
    signals.fundamental_score = score
    signals.score_details = details

    logger.info("%s fundamental score: %d/15 — %s", ticker, score, ", ".join(details) or "no flags")
    return signals

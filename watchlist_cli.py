#!/usr/bin/env python3
"""
Watchlist CLI — add, remove, list, and validate tickers.

Usage:
    python3 watchlist_cli.py list
    python3 watchlist_cli.py add TSLA "Consumer Discretionary" "EV play"
    python3 watchlist_cli.py remove TSLA
    python3 watchlist_cli.py validate
"""

import argparse
import csv
import sys
from pathlib import Path

import yfinance as yf

WATCHLIST_PATH = Path("watchlist.csv")
FIELDNAMES = ["ticker", "sector", "notes"]


def load_watchlist() -> list:
    if not WATCHLIST_PATH.exists():
        return []
    with open(WATCHLIST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def save_watchlist(entries: list) -> None:
    with open(WATCHLIST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(entries)


def cmd_list(args):
    entries = load_watchlist()
    if not entries:
        print("Watchlist is empty.")
        return
    print(f"\n  {'Ticker':<8} {'Sector':<28} {'Notes'}")
    print(f"  {'─' * 60}")
    for e in entries:
        print(f"  {e['ticker']:<8} {e.get('sector', ''):<28} {e.get('notes', '')}")
    print(f"\n  {len(entries)} tickers total.\n")


def cmd_add(args):
    ticker = args.ticker.upper()
    entries = load_watchlist()

    # Check for duplicates
    existing = [e["ticker"].upper() for e in entries]
    if ticker in existing:
        print(f"{ticker} is already in the watchlist.")
        return

    # Validate ticker exists
    try:
        import data_layer
        resolved = data_layer.resolve_ticker(ticker)
        tk = yf.Ticker(resolved)
        info = tk.info or {}
        name = info.get("shortName", info.get("longName", ""))
        if not name and not info.get("regularMarketPrice"):
            print(f"Warning: {ticker} may not be a valid ticker (no data found).")
    except Exception:
        print(f"Warning: Could not validate {ticker}. Adding anyway.")

    entry = {
        "ticker": ticker,
        "sector": args.sector or "",
        "notes": args.notes or "",
    }
    entries.append(entry)
    save_watchlist(entries)
    print(f"Added {ticker} to watchlist ({len(entries)} tickers total).")


def cmd_remove(args):
    ticker = args.ticker.upper()
    entries = load_watchlist()
    original_len = len(entries)

    entries = [e for e in entries if e["ticker"].upper() != ticker]

    if len(entries) == original_len:
        print(f"{ticker} not found in watchlist.")
        return

    save_watchlist(entries)
    print(f"Removed {ticker} from watchlist ({len(entries)} tickers remaining).")


def cmd_validate(args):
    entries = load_watchlist()
    if not entries:
        print("Watchlist is empty.")
        return

    print(f"Validating {len(entries)} tickers...\n")
    valid = 0
    invalid = 0

    for e in entries:
        ticker = e["ticker"]
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d")
            if hist.empty:
                print(f"  {ticker:<8} — NO DATA (may be delisted or invalid)")
                invalid += 1
            else:
                price = hist["Close"].iloc[-1]
                print(f"  {ticker:<8} — OK (${price:,.2f})")
                valid += 1
        except Exception as ex:
            print(f"  {ticker:<8} — ERROR ({ex})")
            invalid += 1

    print(f"\nResults: {valid} valid, {invalid} invalid out of {len(entries)} tickers.")


def main():
    parser = argparse.ArgumentParser(description="Stock Agent Watchlist Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # list
    subparsers.add_parser("list", help="List all tickers in the watchlist")

    # add
    add_parser = subparsers.add_parser("add", help="Add a ticker to the watchlist")
    add_parser.add_argument("ticker", type=str, help="Ticker symbol (e.g. AAPL)")
    add_parser.add_argument("sector", type=str, nargs="?", default="", help="Sector")
    add_parser.add_argument("notes", type=str, nargs="?", default="", help="Notes")

    # remove
    rm_parser = subparsers.add_parser("remove", help="Remove a ticker from the watchlist")
    rm_parser.add_argument("ticker", type=str, help="Ticker symbol to remove")

    # validate
    subparsers.add_parser("validate", help="Validate all tickers have live data")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    commands = {
        "list": cmd_list,
        "add": cmd_add,
        "remove": cmd_remove,
        "validate": cmd_validate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

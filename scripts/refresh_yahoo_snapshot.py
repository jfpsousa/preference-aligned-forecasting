"""
How data/snapshots/yahoo_etf_2018_2023.csv was produced (10 November 2026).

NOT part of reproduce.sh. Yahoo Finance revises adjusted prices retroactively,
so running this today writes a file that differs from the shipped snapshot and
fails the checksum; that is expected. Requires `pip install yfinance`
(snapshot made with yfinance 1.8.0).

    python scripts/refresh_yahoo_snapshot.py --out /tmp/yahoo_new.csv
"""

import argparse

import pandas as pd
import yfinance as yf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    frames = []
    for t in ["SPY", "QQQ", "IWM"]:
        d = yf.download(t, start="2018-01-01", end="2024-01-01", auto_adjust=False,
                        progress=False, actions=False)
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        d = d.reset_index()[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        d.insert(1, "Ticker", t)
        frames.append(d)
    df = pd.concat(frames)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(args.out, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()

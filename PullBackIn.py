#!/usr/bin/env python3
"""
PullBackInTW_balanced.py
Balanced preset for Pullback Buy (平衡設定)
- Default TICKER = "2317.TW"
- Usage: python PullBackInTW_balanced.py --ticker 2317.TW
- Requirements: pip install yfinance pandas numpy twstock
"""

import argparse
import re
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# -------- Strategy parameters (Balanced preset) --------
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_MID = 50
SMA_LONG = 200
VOL_SMA = 20

PULLBACK_NDAYS = 10      # 用最近 N 日高點計算拉回
PULLBACK_PCT = 0.07      # 距離近期高點下跌 >= 7% 視為拉回
VOL_MIN_RATIO_ENTRY = 0.8    # 初次進場允許量 >= 0.8 * 20日均量
VOL_MIN_RATIO_CONFIRM = 1.2  # 回彈確認要求量 >= 1.2 * 20日均量

SL_BUFFER_PCT = 0.015    # 停損緩衝 1.5%

# -------- Helpers for company name --------
def contains_cjk(s: str) -> bool:
    return bool(s and re.search('[\u4e00-\u9fff]', s))

def get_company_name_from_twstock(code_str):
    try:
        import twstock
    except Exception:
        return None
    m = re.search(r'(\d+)', code_str)
    if not m:
        return None
    code = m.group(1)
    info = twstock.codes.get(code)
    if info and hasattr(info, 'name'):
        return info.name
    return None

def get_company_name(ticker):
    # 1) try twstock (offline Taiwan mapping)
    name = get_company_name_from_twstock(ticker)
    if name:
        return name
    # 2) fallback to yfinance info
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        for k in ("shortName", "longName", "name"):
            v = info.get(k)
            if v and isinstance(v, str):
                if contains_cjk(v):
                    return v
                else:
                    fallback = v
        if 'fallback' in locals():
            return f"{fallback} (英文名稱)"
    except Exception:
        pass
    return ticker

# -------- Indicator calculations (Wilder RSI = EWM with alpha=1/n) --------
def calc_sma(series: pd.Series, n: int):
    return series.rolling(n).mean()

def calc_rsi_wilder(close: pd.Series, n: int = 14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = down.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi

def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist

# -------- Data fetch and prepare --------
def fetch_data(ticker: str, days: int = 400):
    # Use yfinance history; auto_adjust=False to use raw prices
    tk = yf.Ticker(ticker)
    df = tk.history(period=f"{days}d", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"取得 {ticker} 資料失敗，請確認代號或網路。")
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    return df

def prepare_df(df: pd.DataFrame):
    df = df.copy()
    df[f"SMA{SMA_SHORT}"] = calc_sma(df['Close'], SMA_SHORT)
    df[f"SMA{SMA_MID}"] = calc_sma(df['Close'], SMA_MID)
    df[f"SMA{SMA_LONG}"] = calc_sma(df['Close'], SMA_LONG)
    df[f"RSI{RSI_PERIOD}"] = calc_rsi_wilder(df['Close'], RSI_PERIOD)
    macd, macd_sig, macd_hist = calc_macd(df['Close'])
    df["MACD"] = macd
    df["MACD_SIG"] = macd_sig
    df["MACD_HIST"] = macd_hist
    df["VOL_SMA"] = df["Volume"].rolling(VOL_SMA).mean()
    return df

# -------- Decision logic: Balanced preset --------
def decision_pullback_balanced(df: pd.DataFrame):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(latest['Close'])
    low = float(latest['Low'])
    high = float(latest['High'])
    vol = float(latest['Volume'])

    sma20 = float(latest[f"SMA{SMA_SHORT}"]) if not pd.isna(latest[f"SMA{SMA_SHORT}"]) else np.nan
    sma50 = float(latest[f"SMA{SMA_MID}"]) if not pd.isna(latest[f"SMA{SMA_MID}"]) else np.nan
    sma200 = float(latest[f"SMA{SMA_LONG}"]) if not pd.isna(latest[f"SMA{SMA_LONG}"]) else np.nan
    rsi = float(latest[f"RSI{RSI_PERIOD}"])
    rsi_prev = float(prev[f"RSI{RSI_PERIOD}"])
    macd = float(latest["MACD"])
    macd_sig = float(latest["MACD_SIG"])
    macd_hist = float(latest["MACD_HIST"])
    vol20 = float(latest["VOL_SMA"]) if not pd.isna(latest["VOL_SMA"]) else np.nan

    recent_high = float(df['High'].iloc[-PULLBACK_NDAYS:].max())
    recent_low = float(df['Low'].iloc[-PULLBACK_NDAYS:].min())
    drop_from_high = (recent_high - low) / recent_high if (recent_high and recent_high > 0) else 0.0

    reasons = []
    flags = {}

    # Trend: SMA50 > SMA200 AND Close > SMA50
    trend_ok = False
    if not np.isnan(sma50) and not np.isnan(sma200):
        trend_ok = (sma50 > sma200) and (close > sma50)
    flags['trend_ok'] = trend_ok
    reasons.append("📈 長期趨勢：SMA50>{:.2f} & Close>{:.2f} → {}".format(sma200, sma50, "✅" if trend_ok else "🔻"))

    # Pullback detection: hit SMA20 OR drop >= PULLBACK_PCT
    pullback_by_sma20 = (not np.isnan(sma20)) and (low <= sma20)
    pullback_by_pct = drop_from_high >= PULLBACK_PCT
    flags['pullback_by_sma20'] = pullback_by_sma20
    flags['pullback_by_pct'] = pullback_by_pct
    if pullback_by_sma20:
        reasons.append(f"🔻 發生拉回：價格觸及/跌破 SMA{SMA_SHORT}（Low {low:.2f} <= SMA{SMA_SHORT} {sma20:.2f}）")
    elif pullback_by_pct:
        reasons.append(f"🔻 發生拉回：距離最近 {PULLBACK_NDAYS} 日高點下跌 {drop_from_high*100:.2f}% ≥ {PULLBACK_PCT*100:.1f}%")
    else:
        reasons.append("ℹ️ 近期未發現合格拉回（未觸及 SMA20 且跌幅不足）")

    # RSI condition: between 30~50 and rising OR crossing 30/40 upward
    rsi_ok = False
    if (30 <= rsi <= 50 and rsi > rsi_prev) or ((rsi_prev < 30 and rsi >= 30) or (rsi_prev < 40 and rsi >= 40)):
        rsi_ok = True
    flags['rsi_ok'] = rsi_ok
    reasons.append(f"🔍 RSI: now {rsi:.2f}, prev {rsi_prev:.2f} → {'✅' if rsi_ok else '🔻'}")

    # MACD: hist rising or macd > signal
    prev_hist = float(df["MACD_HIST"].iloc[-2])
    macd_ok = False
    if (macd_hist > prev_hist and macd_hist > 0) or (macd > macd_sig):
        macd_ok = True
    flags['macd_ok'] = macd_ok
    reasons.append(f"📊 MACD_HIST: now {macd_hist:.6f}, prev {prev_hist:.6f} → {'✅' if macd_ok else '🔻'}")

    # Volume: entry-level allow 0.8*20davg, confirmation requires 1.2*20davg
    vol_entry_ok = (not np.isnan(vol20)) and (vol >= VOL_MIN_RATIO_ENTRY * vol20)
    vol_confirm_ok = (not np.isnan(vol20)) and (vol >= VOL_MIN_RATIO_CONFIRM * vol20)
    flags['vol_entry_ok'] = vol_entry_ok
    flags['vol_confirm_ok'] = vol_confirm_ok
    reasons.append(f"📈 量能：今日 {int(vol)} / 20d avg {int(vol20) if not np.isnan(vol20) else 'N/A'} → entry_ok={'✅' if vol_entry_ok else '🔴'}, confirm_ok={'✅' if vol_confirm_ok else '🔴'}")

    # Final decision logic for Balanced:
    # require: trend_ok AND (pullback_by_sma20 OR pullback_by_pct) AND (rsi_ok OR macd_ok) AND vol_entry_ok
    entry = False
    if trend_ok and (pullback_by_sma20 or pullback_by_pct) and (rsi_ok or macd_ok) and vol_entry_ok:
        entry = True
    flags['entry'] = entry

    # Suggested buy zone & stop loss (conservative)
    buy_zone = None
    buy_break = None
    stop_loss = None
    if pullback_by_sma20 and not np.isnan(sma20):
        # conservative zone between SMA20*0.99 and max(close,sma20)
        low_zone = max(recent_low, sma20 * 0.98)
        high_zone = max(close, sma20)
        buy_zone = (low_zone, high_zone)
        stop_loss = recent_low * (1 - SL_BUFFER_PCT)
    elif pullback_by_pct:
        buy_zone = (recent_low, (recent_high + recent_low) / 2.0)
        stop_loss = recent_low * (1 - SL_BUFFER_PCT)
    else:
        # no clear pullback -> optional breakout entry
        buy_break = high * 1.002
        stop_loss = recent_low * (1 - SL_BUFFER_PCT)

    # Collect values for printing / logging
    values = {
        "close": close,
        "low": low,
        "high": high,
        "volume": vol,
        "SMA20": sma20,
        "SMA50": sma50,
        "SMA200": sma200,
        "RSI": rsi,
        "RSI_prev": rsi_prev,
        "MACD": macd,
        "MACD_SIG": macd_sig,
        "MACD_HIST": macd_hist,
        "VOL20": vol20,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "drop_from_high_pct": drop_from_high * 100,
    }

    plan = {
        "buy_zone": buy_zone,
        "buy_break": buy_break,
        "stop_loss": stop_loss,
        "vol_entry_ok": vol_entry_ok,
        "vol_confirm_ok": vol_confirm_ok,
        "sl_buffer_pct": SL_BUFFER_PCT,
    }

    return {"entry": entry, "reasons": reasons, "flags": flags, "values": values, "plan": plan}

# -------- Pretty print (emoji-friendly) --------
def pretty_print(result: dict, ticker: str, company_name: str):
    vals = result["values"]
    plan = result["plan"]
    print("=======================================================")
    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📌 標的：{ticker}  —  {company_name}")
    print("-------------------------------------------------------")
    print(f"💰 今日收盤：{vals['close']:.2f}    🔽 今日最低：{vals['low']:.2f}    🔼 今日最高：{vals['high']:.2f}")
    print(f"📈 今日成交量：{int(vals['volume'])}    （20日平均量：{int(vals['VOL20']) if not np.isnan(vals['VOL20']) else 'N/A'}）")
    print("-------------------------------------------------------")
    print(f"📊 SMA{SMA_SHORT}：{vals['SMA20']:.2f}    SMA{SMA_MID}：{vals['SMA50']:.2f}    SMA{SMA_LONG}：{vals['SMA200']:.2f}")
    print(f"📉 RSI{RSI_PERIOD}：{vals['RSI']:.2f}（前：{vals['RSI_prev']:.2f}）")
    print(f"📈 MACD：{vals['MACD']:.4f}    SIG：{vals['MACD_SIG']:.4f}    HIST：{vals['MACD_HIST']:.6f}")
    print("-------------------------------------------------------")
    print(f"📅 最近 {PULLBACK_NDAYS} 日高點：{vals['recent_high']:.2f}    低點：{vals['recent_low']:.2f}")
    print(f"🔻 距離最近高點跌幅：{vals['drop_from_high_pct']:.2f}%")
    print("=======================================================")
    print("🔎 判斷結果（Balanced preset）：")
    print(" ➤ 是否為合格拉回買：", "✅ ✅ ✅ 合格（可考慮分批進場）" if result["entry"] else "❌ ❌ ❌ 不建議拉回買（不符條件）")
    print("\n📝 觸發/檢核細項：")
    for r in result["reasons"]:
        print("  -", r)
    print("\n🎯 建議進場計畫（僅供參考）：")
    if plan["buy_zone"] is not None:
        lo, hi = plan["buy_zone"]
        print(f"  🟢 建議分批買入區間（保守參考）：{lo:.2f} ~ {hi:.2f}")
    if plan["buy_break"] is not None:
        print(f"  🔵 或等突破買進（突破當日高點）：{plan['buy_break']:.2f}")
    if plan["stop_loss"] is not None:
        print(f"  🛑 建議停損：{plan['stop_loss']:.2f}（最近 {PULLBACK_NDAYS} 日低點下方 {plan.get('sl_buffer_pct')*100:.2f}%）")
    print("-------------------------------------------------------")
    print("💡 小建議：首次進場建議分批（例如 40% / 30% / 30%），回彈放量再加碼。")
    print("=======================================================")

# -------- Main --------
def main():
    parser = argparse.ArgumentParser(description="PullBackInTW (Balanced preset)")
    parser.add_argument("--ticker", type=str, default="2317.TW", help="Ticker (e.g., 2317.TW or NVDA)")
    parser.add_argument("--days", type=int, default=400, help="Days history to fetch")
    args = parser.parse_args()

    ticker = args.ticker
    days = args.days

    df = fetch_data(ticker, days)
    df = prepare_df(df)
    if df.shape[0] < max(SMA_LONG, RSI_PERIOD, VOL_SMA, PULLBACK_NDAYS) + 5:
        raise RuntimeError("資料筆數不足，請增加 days 或確認資料來源。")

    result = decision_pullback_balanced(df)
    company_name = get_company_name(ticker)
    pretty_print(result, ticker, company_name)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("執行發生錯誤：", e)
        print("檢查：是否安裝 yfinance/pandas/numpy (twstock 可選)、或 ticker 格式是否正確（TW 市場加 .TW）")

"""
鴻海 (2317.TW) 短線進場判斷範例程式
- 需求：列印今日 Close、Volume、指標值，並判斷是否可進場與原因
- 指標：SMA20, SMA50, RSI14, MACD (12,26,9), Volume 20-day avg
- 依賴套件：yfinance, pandas, numpy
    pip install yfinance pandas numpy
- 使用方式：python fh_short_entry.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

TICKER = "2317.TW"   # 鴻海（台灣）
DAYS = 300           # 抓取天數
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_LONG = 50
VOL_SMA = 20

def fetch_data(ticker, days):
    df = yf.Ticker(ticker).history(period=f"{days}d")
    if df.empty:
        raise RuntimeError("取得資料失敗，請確認網路與代號是否正確。")
    df = df[['Open','High','Low','Close','Volume']]
    df.index = pd.to_datetime(df.index)
    return df

def calc_sma(df, n):
    return df['Close'].rolling(n).mean()

def calc_rsi(df, n=14):
    # Wilder's RSI (EMA smoothing with alpha=1/n)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use exponential weighted mean with alpha=1/n (Wilder)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # 初期值填 50（中性）
    return rsi

def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist

def decision_logic(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = latest['Close']
    vol = latest['Volume']

    sma20 = latest[f"SMA{SMA_SHORT}"]
    sma50 = latest[f"SMA{SMA_LONG}"]
    rsi = latest[f"RSI{RSI_PERIOD}"]
    rsi_prev = prev[f"RSI{RSI_PERIOD}"]
    macd = latest["MACD"]
    macd_sig = latest["MACD_SIG"]
    macd_hist = latest["MACD_HIST"]
    vol20 = latest["VOL_SMA"]

    reasons = []
    entry = False

    # 判斷條件（可調）
    # A) 趨勢：價格在短期均線上，且短期均線高於長期均線（上升趨勢）
    trend_ok = (close > sma20) and (sma20 > sma50)
    if trend_ok:
        reasons.append("價格站上 SMA20，且 SMA20 > SMA50（上升趨勢）")
    else:
        reasons.append("趨勢不明或偏空（價格未站上 SMA20 或 SMA20 <= SMA50）")

    # B) RSI 反彈訊號：RSI 向上，且穿越 40 或 30（可視為動能回復）
    rsi_signal = False
    if (rsi_prev < 40 and rsi >= 40) or (rsi_prev < 30 and rsi >= 30):
        rsi_signal = True
        reasons.append("RSI 發生回升且穿越關鍵位（30/40），短線動能回復")
    elif rsi > rsi_prev and rsi > 45:
        rsi_signal = True
        reasons.append("RSI 上升且位於中高位（>45），動能偏多")
    else:
        reasons.append("RSI 未明顯回升或仍偏弱")

    # C) MACD 訊號：macd 線向上且 hist > 0 或剛剛黃金交叉
    macd_signal = False
    # macd 正且 histogram 正，或由下往上穿過 signal
    if (macd_hist > 0) or ( (df.iloc[-2]["MACD"] < df.iloc[-2]["MACD_SIG"]) and (macd > macd_sig) ):
        macd_signal = True
        reasons.append("MACD 呈多頭（hist > 0 或剛形成上穿）")
    else:
        reasons.append("MACD 未形成明顯多頭")

    # D) 成交量確認：今日量 >= 1.2 * 20日平均量
    vol_ok = vol >= 1.2 * vol20
    if vol_ok:
        reasons.append("今天成交量放大（> 20 日均量 * 1.2），訊號可信度提高")
    else:
        reasons.append("成交量未放大（可能缺乏承接）")

    # 最終進場判斷（可自行調整組合邏輯）
    # 目前採：趨勢 + (RSI 或 MACD) + 成交量確認
    if trend_ok and (rsi_signal or macd_signal) and vol_ok:
        entry = True
    else:
        entry = False

    return {
        "entry": entry,
        "reasons": reasons,
        "values": {
            "close": close,
            "volume": vol,
            "SMA20": sma20,
            "SMA50": sma50,
            "RSI": rsi,
            "RSI_prev": rsi_prev,
            "MACD": macd,
            "MACD_SIG": macd_sig,
            "MACD_HIST": macd_hist,
            "VOL20": vol20
        }
    }

def prepare_df(df):
    df[f"SMA{SMA_SHORT}"] = calc_sma(df, SMA_SHORT)
    df[f"SMA{SMA_LONG}"] = calc_sma(df, SMA_LONG)
    df[f"RSI{RSI_PERIOD}"] = calc_rsi(df, RSI_PERIOD)
    macd, macd_sig, macd_hist = calc_macd(df)
    df["MACD"] = macd
    df["MACD_SIG"] = macd_sig
    df["MACD_HIST"] = macd_hist
    df["VOL_SMA"] = df["Volume"].rolling(VOL_SMA).mean()
    return df

def pretty_print(result):
    vals = result["values"]
    print("=== 鴻海 (2317.TW) 短線進場判斷 ===")
    print("時間：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"今日收盤價：{vals['close']:.2f}")
    print(f"今日成交量：{int(vals['volume'])}")
    print(f"SMA{SMA_SHORT}：{vals['SMA20']:.2f}    SMA{SMA_LONG}：{vals['SMA50']:.2f}")
    print(f"RSI{RSI_PERIOD}：{vals['RSI']:.2f}（前一日：{vals['RSI_prev']:.2f}）")
    print(f"MACD：{vals['MACD']:.4f}    MACD_SIG：{vals['MACD_SIG']:.4f}    HIST：{vals['MACD_HIST']:.6f}")
    print(f"20日平均量：{int(vals['VOL20'])}")
    print("--------------------------------------")
    print("是否可進場：", "✅ 可以進場" if result["entry"] else "❌ 不建議進場")
    print("判斷理由：")
    for r in result["reasons"]:
        print(" -", r)
    print("\n（提示：可修改邏輯門檻，例如 RSI 閾值、成交量放大倍數、以及是否強制要求趨勢）")

def main():
    df = fetch_data(TICKER, DAYS)
    df = prepare_df(df)
    # 檢查是否資料足夠
    if df.shape[0] < max(SMA_LONG, RSI_PERIOD, VOL_SMA) + 5:
        raise RuntimeError("資料筆數不足，請增加 DAYS 或確認資料來源。")
    res = decision_logic(df)
    pretty_print(res)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("執行發生錯誤：", e)
        print("請確認已安裝套件並有網路，或代號是否正確（TW 市場需加 .TW）")

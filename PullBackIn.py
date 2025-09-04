"""
拉回買示範程式（鴻海 2317.TW）
- 指標：SMA20, SMA50, SMA200, RSI14, MACD(12,26,9), 20日平均量
- 拉回買邏輯（預設）：
  1) 長期趨勢為多頭：SMA50 > SMA200 且 價格 > SMA50
  2) 發生短期拉回：今日收盤或最低價低於 SMA20 或 距離最近 N 日高點下跌 >= pullback_pct
  3) RSI 在 30~50 並有回升跡象（或穿越 30/40）
  4) MACD 多頭或 histogram 開始回升
  5) 成交量在拉回底部時不過度放大（反而期待回升時量能跟上）——這邊要求今天量 >= 0.8 * 20日均量（可調）
  若以上多數條件成立，視為「合格拉回買」，並給出建議進場區間與停損位置
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

TICKER = "2317.TW"
DAYS = 400

# 指標參數（可調）
RSI_PERIOD = 14
SMA_SHORT = 20
SMA_MID = 50
SMA_LONG = 200
VOL_SMA = 20
PULLBACK_NDAYS = 10        # 最近高點參考天數
PULLBACK_PCT = 0.07        # 下跌 >= 7% 視為拉回
VOL_MIN_RATIO = 0.8        # 今天量 >= VOL_MIN_RATIO * 20日均量 -> 視為量能可接受

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
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    return rsi

def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_sig
    return macd, macd_sig, macd_hist

def prepare_df(df):
    df[f"SMA{SMA_SHORT}"] = calc_sma(df, SMA_SHORT)
    df[f"SMA{SMA_MID}"] = calc_sma(df, SMA_MID)
    df[f"SMA{SMA_LONG}"] = calc_sma(df, SMA_LONG)
    df[f"RSI{RSI_PERIOD}"] = calc_rsi(df, RSI_PERIOD)
    macd, macd_sig, macd_hist = calc_macd(df)
    df["MACD"] = macd
    df["MACD_SIG"] = macd_sig
    df["MACD_HIST"] = macd_hist
    df["VOL_SMA"] = df["Volume"].rolling(VOL_SMA).mean()
    return df

def decision_pullback(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    close = latest['Close']
    low = latest['Low']
    vol = latest['Volume']
    sma20 = latest[f"SMA{SMA_SHORT}"]
    sma50 = latest[f"SMA{SMA_MID}"]
    sma200 = latest[f"SMA{SMA_LONG}"]
    rsi = latest[f"RSI{RSI_PERIOD}"]
    rsi_prev = prev[f"RSI{RSI_PERIOD}"]
    macd = latest["MACD"]
    macd_sig = latest["MACD_SIG"]
    macd_hist = latest["MACD_HIST"]
    vol20 = latest["VOL_SMA"]

    # 最近 N 日高低
    recent_high = df['High'].iloc[-PULLBACK_NDAYS:].max()
    recent_low = df['Low'].iloc[-PULLBACK_NDAYS:].min()
    drop_from_high = (recent_high - low) / recent_high if recent_high>0 else 0.0

    reasons = []
    entry = False

    # 1) 長期趨勢：SMA50 > SMA200 且 價格 > SMA50
    long_trend = (sma50 is not None and sma200 is not None and sma50 > sma200) and (close > sma50)
    if long_trend:
        reasons.append("長期趨勢為多頭（SMA50 > SMA200 且 價格 > SMA50）")
    else:
        reasons.append("長期趨勢非典型多頭（SMA50 <= SMA200 或 價格 <= SMA50）")

    # 2) 是否處於拉回：低於 SMA20 或 距離最近高點下跌達門檻
    pullback_by_sma20 = low <= sma20 if not pd.isna(sma20) else False
    pullback_by_pct = drop_from_high >= PULLBACK_PCT
    if pullback_by_sma20:
        reasons.append(f"發生拉回：價格觸及或跌破 SMA{SMA_SHORT}（Low {low:.2f} <= SMA{SMA_SHORT} {sma20:.2f}）")
    elif pullback_by_pct:
        reasons.append(f"發生拉回：距離最近 {PULLBACK_NDAYS} 日高點下跌 {drop_from_high*100:.2f}% >= {PULLBACK_PCT*100:.1f}%")
    else:
        reasons.append("近期尚無明顯拉回（未跌破 SMA20 且距離近期高點跌幅不足）")

    # 3) RSI 條件：RSI 在 30~50，且有向上跡象（或穿越 30/40）
    rsi_ok = False
    if (30 <= rsi <= 50 and rsi > rsi_prev) or ((rsi_prev < 30 and rsi >= 30) or (rsi_prev < 40 and rsi >= 40)):
        rsi_ok = True
        reasons.append(f"RSI 處於回補區並向上（RSI={rsi:.2f}，前一日 {rsi_prev:.2f}）")
    else:
        reasons.append(f"RSI 未明顯回升或不在理想回補區（RSI={rsi:.2f}）")

    # 4) MACD 條件：Hist 開始回升或 MACD > signal
    macd_ok = False
    prev_hist = df["MACD_HIST"].iloc[-2]
    if (macd_hist > prev_hist and macd_hist > 0) or (macd > macd_sig):
        macd_ok = True
        reasons.append("MACD 顯示動能回復（hist 開始上升或 MACD > signal）")
    else:
        reasons.append("MACD 動能尚未明顯回復")

    # 5) 成交量：今天量不過小（避免成交量完全萎縮）; 同時拉回底部最好量縮，反彈量增 -- 這裡簡化為今日量 >= VOL_MIN_RATIO * 20日均量
    vol_ok = (not pd.isna(vol20)) and (vol >= VOL_MIN_RATIO * vol20)
    if vol_ok:
        reasons.append(f"今日量 {int(vol)} >= {VOL_MIN_RATIO} * 20日均量 ({int(vol20)})，量能可接受")
    else:
        reasons.append(f"今日量 {int(vol)} < {VOL_MIN_RATIO} * 20日均量 ({int(vol20)})，量能偏弱")

    # 最終判斷邏輯：長期趨勢 + (拉回 by sma20 或 by pct) + (RSI 或 MACD) + vol_ok
    if long_trend and (pullback_by_sma20 or pullback_by_pct) and (rsi_ok or macd_ok) and vol_ok:
        entry = True
    else:
        entry = False

    # 建議進場區間（保守）：介於今日收盤和最近低點到 SMA20 之間，或直接建議以突破當日高點買進
    # 這裡給兩種選項供參考
    buy_zone = None
    buy_break = None
    stop_loss = None
    # 建議停損：最近 N 日低點下方一定比例（例如 1.5%）或跌破 SMA50
    SL_BUFFER_PCT = 0.015
    if pullback_by_sma20 and not pd.isna(sma20):
        buy_zone = (max(low, sma20*0.98), max(close, sma20))  # 保守：0.98*sma20 ~ sma20 ~ close
        stop_loss = recent_low * (1 - SL_BUFFER_PCT)
    elif pullback_by_pct:
        buy_zone = (recent_low, (recent_high + recent_low)/2)  # 用最近低點到中間價做參考
        stop_loss = recent_low * (1 - SL_BUFFER_PCT)
    else:
        # 非拉回，可使用突破買法（可選）
        buy_break = latest['High'] * 1.002  # 若想用突破買進
        stop_loss = recent_low * (1 - SL_BUFFER_PCT)

    return {
        "entry": entry,
        "reasons": reasons,
        "values": {
            "close": close,
            "low": low,
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
            "drop_from_high_pct": drop_from_high * 100
        },
        "plan": {
            "buy_zone": buy_zone,
            "buy_break": buy_break,
            "stop_loss": stop_loss
        }
    }

def pretty_print(res):
    vals = res["values"]
    plan = res["plan"]
    print("=== 鴻海 (2317.TW) — 拉回買判斷 ===")
    print("時間：", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"今日收盤：{vals['close']:.2f}    今日最低：{vals['low']:.2f}    今日成交量：{int(vals['volume'])}")
    print(f"SMA{SMA_SHORT}：{vals['SMA20']:.2f}    SMA{SMA_MID}：{vals['SMA50']:.2f}    SMA{SMA_LONG}：{vals['SMA200']:.2f}")
    print(f"RSI{RSI_PERIOD}：{vals['RSI']:.2f}（前：{vals['RSI_prev']:.2f}）")
    print(f"MACD：{vals['MACD']:.4f}    SIG：{vals['MACD_SIG']:.4f}    HIST：{vals['MACD_HIST']:.6f}")
    print(f"{PULLBACK_NDAYS} 日高點：{vals['recent_high']:.2f}    {PULLBACK_NDAYS} 日低點：{vals['recent_low']:.2f}")
    print(f"距離最近高點跌幅：{vals['drop_from_high_pct']:.2f}%")
    print(f"20日平均量：{int(vals['VOL20'])}")
    print("--------------------------------------")
    print("是否為合格拉回買：", "✅ 合格，可考慮分批進場" if res["entry"] else "❌ 不建議拉回買（不符條件）")
    print("\n判斷理由：")
    for r in res["reasons"]:
        print(" -", r)
    print("\n建議進場計畫（僅供參考）：")
    if plan["buy_zone"] is not None:
        lo, hi = plan["buy_zone"]
        print(f" - 建議分批買入區間（保守參考）：{lo:.2f} ~ {hi:.2f}")
    if plan["buy_break"] is not None:
        print(f" - 或等突破買進（突破當日高點）買點：{plan['buy_break']:.2f}")
    if plan["stop_loss"] is not None:
        print(f" - 建議停損：{plan['stop_loss']:.2f}（例如最近 {PULLBACK_NDAYS} 日低點下方 {SL_BUFFER_PCT*100:.2f}%）")
    print("\n提示：可調整參數包括：PULLBACK_PCT、PULLBACK_NDAYS、VOL_MIN_RATIO、RSI 閾值等。")

def main():
    df = fetch_data(TICKER, DAYS)
    df = prepare_df(df)
    if df.shape[0] < max(SMA_LONG, RSI_PERIOD, VOL_SMA, PULLBACK_NDAYS) + 5:
        raise RuntimeError("資料筆數不足，請增加 DAYS 或確認資料來源。")
    res = decision_pullback(df)
    pretty_print(res)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("執行發生錯誤：", e)
        print("請確認已安裝套件並有網路，或代號是否正確（TW 市場需加 .TW）")

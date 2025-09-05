"""
拉回買示範程式（鴻海 2317.TW）- 帶 emoji 並自動抓中文公司名稱
- pip install yfinance pandas numpy
- 建議：若要更好中文名辨識，另外安裝 twstock -> pip install twstock
- python pullback_entry_with_name.py
"""
import yfinance as yf
import pandas as pd
import numpy as np
import re
from datetime import datetime

TICKER = "2317.TW"   # 可改成其他，例如 "2330.TW"
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

# 停損 buffer（全域）
SL_BUFFER_PCT = 0.015      # 停損建議比最近低點再下方 1.5%

def get_company_name_from_twstock(code_str):
    """
    嘗試用 twstock 取得中文名稱（若系統有安裝 twstock）
    code_str: '2317' 或 '2317.TW' 均可
    回傳中文名稱或 None
    """
    try:
        import twstock
    except Exception:
        return None
    # 取出數字代碼
    m = re.search(r'(\d+)', code_str)
    if not m:
        return None
    code = m.group(1)
    info = twstock.codes.get(code)
    if info and hasattr(info, 'name'):
        return info.name
    return None

def contains_cjk(s: str) -> bool:
    """檢查字串是否含有中文字元（CJK）"""
    if not s:
        return False
    return bool(re.search('[\u4e00-\u9fff]', s))

def get_company_name(ticker):
    """
    取得公司中文名稱的策略：
    1) 優先用 twstock（若安裝且有對應）
    2) 再用 yfinance .info 的 longName/shortName，若包含中文就採用
    3) 若只有英文則回傳英文（並標示為 fallback）
    """
    # 1) twstock
    name = get_company_name_from_twstock(ticker)
    if name:
        return name

    # 2) yfinance
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        for k in ("shortName", "longName", "name"):
            v = info.get(k)
            if v and isinstance(v, str):
                if contains_cjk(v):
                    return v  # 有中文，直接回傳
                else:
                    # 暫存英文名稱作為 fallback
                    fallback = v
        # 如果找不到中文，但有英文 fallback，回傳並標示
        if 'fallback' in locals():
            return f"{fallback} (英文名稱，無中文資料)"
    except Exception:
        pass

    # 3) 最後回傳 ticker 自己（若一律找不到）
    return ticker

# -------- 以下為原本的指標與判斷程式（略作整合） --------
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
    drop_from_high = (recent_high - low) / recent_high if (recent_high and recent_high>0) else 0.0

    reasons = []
    entry = False

    # 1) 長期趨勢：SMA50 > SMA200 且 價格 > SMA50
    long_trend = (not pd.isna(sma50) and not pd.isna(sma200) and sma50 > sma200) and (close > sma50)
    if long_trend:
        reasons.append("📈 長期趨勢為多頭（SMA50 > SMA200 且 價格 > SMA50）")
    else:
        reasons.append("🔻 長期趨勢非典型多頭（SMA50 <= SMA200 或 價格 <= SMA50）")

    # 2) 是否處於拉回：低於 SMA20 或 距離最近高點下跌達門檻
    pullback_by_sma20 = (not pd.isna(sma20)) and (low <= sma20)
    pullback_by_pct = drop_from_high >= PULLBACK_PCT
    if pullback_by_sma20:
        reasons.append(f"🔻 發生拉回：價格觸及或跌破 SMA{SMA_SHORT}（Low {low:.2f} <= SMA{SMA_SHORT} {sma20:.2f}）")
    elif pullback_by_pct:
        reasons.append(f"🔻 發生拉回：距離最近 {PULLBACK_NDAYS} 日高點下跌 {drop_from_high*100:.2f}% ≥ {PULLBACK_PCT*100:.1f}%")
    else:
        reasons.append("ℹ️ 近期尚無明顯拉回（未跌破 SMA20 且距離近期高點跌幅不足）")

    # 3) RSI 條件
    rsi_ok = False
    if (30 <= rsi <= 50 and rsi > rsi_prev) or ((rsi_prev < 30 and rsi >= 30) or (rsi_prev < 40 and rsi >= 40)):
        rsi_ok = True
        reasons.append(f"🔍 RSI 回補並向上（RSI={rsi:.2f}，前：{rsi_prev:.2f}）")
    else:
        reasons.append(f"🔍 RSI 未明顯回升或不在理想回補區（RSI={rsi:.2f}）")

    # 4) MACD 條件
    macd_ok = False
    prev_hist = df["MACD_HIST"].iloc[-2]
    if (macd_hist > prev_hist and macd_hist > 0) or (macd > macd_sig):
        macd_ok = True
        reasons.append("📊 MACD 顯示動能回復（hist 開始上升或 MACD > signal）")
    else:
        reasons.append("📊 MACD 動能尚未明顯回復")

    # 5) 成交量：今天量不過小
    vol_ok = (not pd.isna(vol20)) and (vol >= VOL_MIN_RATIO * vol20)
    if vol_ok:
        reasons.append(f"🟢 量能：今日量 {int(vol)} ≥ {VOL_MIN_RATIO} * 20日均量 ({int(vol20)})")
    else:
        reasons.append(f"🔴 量能：今日量 {int(vol)} < {VOL_MIN_RATIO} * 20日均量 ({int(vol20)})")

    # 最終判斷
    if long_trend and (pullback_by_sma20 or pullback_by_pct) and (rsi_ok or macd_ok) and vol_ok:
        entry = True
    else:
        entry = False

    # 建議進場區間與停損
    buy_zone = None
    buy_break = None
    stop_loss = None

    if pullback_by_sma20 and not pd.isna(sma20):
        buy_zone = (max(low, sma20*0.98), max(close, sma20))
        stop_loss = recent_low * (1 - SL_BUFFER_PCT)
    elif pullback_by_pct:
        buy_zone = (recent_low, (recent_high + recent_low)/2)
        stop_loss = recent_low * (1 - SL_BUFFER_PCT)
    else:
        buy_break = latest['High'] * 1.002
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
            "stop_loss": stop_loss,
            "sl_buffer_pct": SL_BUFFER_PCT
        }
    }

def pretty_print(res, ticker):
    # 取得公司中文名稱（或備援名稱）
    company_name = get_company_name(ticker)
    vals = res["values"]
    plan = res["plan"]
    print("===============================================")
    print(f"🕒 時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📌 標的：{ticker}  —  {company_name}")
    print("-----------------------------------------------")
    print(f"💰 今日收盤：{vals['close']:.2f}    🔽 今日最低：{vals['low']:.2f}")
    print(f"📈 今日成交量：{int(vals['volume'])}    （20日平均量：{int(vals['VOL20'])}）")
    print("-----------------------------------------------")
    print(f"📊 SMA{SMA_SHORT}：{vals['SMA20']:.2f}    SMA{SMA_MID}：{vals['SMA50']:.2f}    SMA{SMA_LONG}：{vals['SMA200']:.2f}")
    print(f"📉 RSI{RSI_PERIOD}：{vals['RSI']:.2f}（前：{vals['RSI_prev']:.2f}）")
    print(f"📈 MACD：{vals['MACD']:.4f}    SIG：{vals['MACD_SIG']:.4f}    HIST：{vals['MACD_HIST']:.6f}")
    print("-----------------------------------------------")
    print(f"📅 最近 {PULLBACK_NDAYS} 日高點：{vals['recent_high']:.2f}    低點：{vals['recent_low']:.2f}")
    print(f"🔻 距離最近高點跌幅：{vals['drop_from_high_pct']:.2f}%")
    print("===============================================")
    print("🔎 判斷結果：")
    print(" ➤ 是否為合格拉回買：", "✅ ✅ ✅ 合格，可考慮分批進場" if res["entry"] else "❌ ❌ ❌ 不建議拉回買（不符條件）")
    print("\n📝 判斷理由：")
    for r in res["reasons"]:
        print("  -", r)
    print("\n🎯 建議進場計畫（僅供參考）：")
    if plan["buy_zone"] is not None:
        lo, hi = plan["buy_zone"]
        print(f"  🟢 建議分批買入區間（保守參考）：{lo:.2f} ~ {hi:.2f}")
    if plan["buy_break"] is not None:
        print(f"  🔵 或等突破買進（突破當日高點）買點：{plan['buy_break']:.2f}")
    if plan["stop_loss"] is not None:
        print(f"  🛑 建議停損：{plan['stop_loss']:.2f}（最近 {PULLBACK_NDAYS} 日低點下方 {plan.get('sl_buffer_pct', SL_BUFFER_PCT)*100:.2f}%）")
    print("===============================================")
    print("💡 提示：若要調整圖示或文字，打開 pretty_print() 修改 emoji 或字串即可。")

def main():
    df = fetch_data(TICKER, DAYS)
    df = prepare_df(df)
    if df.shape[0] < max(SMA_LONG, RSI_PERIOD, VOL_SMA, PULLBACK_NDAYS) + 5:
        raise RuntimeError("資料筆數不足，請增加 DAYS 或確認資料來源。")
    res = decision_pullback(df)
    pretty_print(res, TICKER)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("執行發生錯誤：", e)
        print("請確認已安裝套件並有網路，或代號是否正確（TW 市場需加 .TW）")

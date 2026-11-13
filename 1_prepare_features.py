import pandas as pd
import numpy as np

# 1. Load the clean 5-minute Bank NIFTY dataset
df = pd.read_csv("banknifty_clean_5m.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df["Date"] = df["Datetime"].dt.date
df["Time_Str"] = df["Datetime"].dt.strftime("%H:%M")

# 2. Calculate Daily-Resetting VWAP (Indicator 1)
df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
df["TP_Vol"] = df["TP"] * df["Volume"]
df["VWAP"] = df.groupby("Date")["TP_Vol"].cumsum() / df.groupby("Date")["Volume"].cumsum()
df["VWAP_Dist"] = (df["Close"] - df["VWAP"]) / df["VWAP"]
df["VWAP_Slope"] = df["VWAP"].diff()

# 3. Calculate 20-period Bollinger Bands (Indicator 2)
n, k = 20, 1.5
df["MA20"] = df["Close"].rolling(n).mean()
df["STD20"] = df["Close"].rolling(n).std()
df["Upper_Band"] = df["MA20"] + (k * df["STD20"])
df["Lower_Band"] = df["MA20"] - (k * df["STD20"])
df["Z_Score"] = (df["Close"] - df["MA20"]) / df["STD20"]
df["BW"] = (df["Upper_Band"] - df["Lower_Band"]) / df["MA20"]
df["BW_Change"] = df["BW"].diff()

# 4. Helper context features
df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
df["Ret_1bar"] = df["Close"].pct_change()

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv("dataset_features.csv", index=False)
print("Step 1 Complete! Saved dataset_features.csv with", len(df), "rows.")
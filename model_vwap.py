import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

print("1. Starting script...")

# 1. Load your dataset
try:
    df = pd.read_csv('banknifty_5min_data.csv')
    print(f"2. Dataset loaded successfully! Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: Could not find 'banknifty_5min_data.csv'.")
    exit()

# 2. Ensure timestamps and sorting are correct
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 3. Calculate VWAP (Resets daily)
df['TP'] = (df['high'] + df['low'] + df['close']) / 3
df['date'] = df['timestamp'].dt.date
df['TP_V'] = df['TP'] * df['volume']
df['cum_TP_V'] = df.groupby('date')['TP_V'].cumsum()
df['cum_V'] = df.groupby('date')['volume'].cumsum()
df['VWAP'] = df['cum_TP_V'] / df['cum_V']
print("3. VWAP calculated.")

# 4. Feature Engineering
df['vwap_distance'] = (df['close'] - df['VWAP']) / df['VWAP']
df['vol_ma20'] = df['volume'].rolling(window=3).mean()
df['volume_ratio'] = df['volume'] / df['vol_ma20']
df['bar_return'] = df['close'].pct_change()
df = df.dropna().reset_index(drop=True)
print(f"4. Features engineered. Rows left after cleaning: {len(df)}")

# 5. Define Target Labels
def get_target(sub_df, window=3, target_pct=0.005, stop_pct=0.003):
    labels = []
    closes = sub_df['close'].values
    for i in range(len(closes) - window):
        entry_price = closes[i]
        future_prices = closes[i+1 : i+1+window]
        hit_target = np.any(future_prices >= entry_price * (1 + target_pct))
        hit_stop = np.any(future_prices <= entry_price * (1 - stop_pct))
        if hit_target and not hit_stop:
            labels.append(1)
        else:
            labels.append(0)
    labels.extend([0] * window)
    return labels

df['target'] = get_target(df)

# 6. Train/Test Split & Random Forest Model
feature_cols = ['vwap_distance', 'volume_ratio', 'bar_return']
X = df[feature_cols]
y = df['target']

split_idx = int(len(df) * 0.8)
if split_idx == 0:
    split_idx = 1

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(X_train, y_train)

# 7. Evaluation & Financial Metrics Calculation
if len(X_test) > 0:
    preds = rf_model.predict_proba(X_test)
    y_pred_proba = preds[:, 1] if preds.shape[1] > 1 else preds[:, 0]
else:
    y_pred_proba = []

y_pred = (y_pred_proba >= 0.5).astype(int) if len(y_pred_proba) > 0 else []

# Classification Metrics
accuracy = accuracy_score(y_test, y_pred) if len(y_test) > 0 else 0
precision = precision_score(y_test, y_pred, zero_division=0) if len(y_test) > 0 else 0
recall = recall_score(y_test, y_pred, zero_division=0) if len(y_test) > 0 else 0
f1 = f1_score(y_test, y_pred, zero_division=0) if len(y_test) > 0 else 0

# Financial Backtest Metrics
if len(y_pred) > 0:
    strat_returns = y_pred * X_test['bar_return'].values
    cumulative_returns = np.cumprod(1 + strat_returns) - 1
    net_profit = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    
    # Sharpe Ratio
    if len(strat_returns) > 1 and np.std(strat_returns) > 0:
        sharpe_ratio = (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252 * 75)
    else:
        sharpe_ratio = 0.0
        
    # Maximum Drawdown
    peak = np.maximum.accumulate(1 + cumulative_returns) if len(cumulative_returns) > 0 else np.array([1])
    drawdown = (1 + cumulative_returns) / peak - 1
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
else:
    net_profit = 0.0
    sharpe_ratio = 0.0
    max_drawdown = 0.0

print("\n==========================================")
print("--- Model 1 (VWAP + RF) Performance ---")
print("==========================================")
print(f"Accuracy Rate  : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1-Score       : {f1:.4f}")
print("------------------------------------------")
print(f"Net Profit     : {net_profit * 100:.2f}%")
print(f"Sharpe Ratio   : {sharpe_ratio:.4f}")
print(f"Max Drawdown   : {max_drawdown * 100:.2f}%")
print("==========================================")

# 8. Plot Cumulative Returns using Matplotlib
if len(y_pred) > 0:
    plt.figure(figsize=(10, 5))
    plot_timestamps = df['timestamp'].iloc[split_idx : split_idx + len(cumulative_returns)]
    plt.plot(plot_timestamps, cumulative_returns * 100, label='VWAP + RF Strategy', color='blue', linewidth=2)
    plt.title('Model 1 Cumulative Return (%) - Bank NIFTY Intraday')
    plt.xlabel('Timestamp')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    print("\nDisplaying strategy performance plot window...")
    plt.show()
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. SETTINGS & UI ---
st.set_page_config(page_title="Stock Fund Analyzer", layout="wide")

# Professional Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    .decision-box { padding: 30px; border-radius: 15px; text-align: center; font-weight: bold; margin: 20px 0; border: 2px solid; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🤖 Model Configuration")
    ticker = st.selectbox("Select Asset", 
        ["SPY", "QQQ", "DIA", "AAPL", "MSFT", "TSLA", "GLD", "NVDA", "GOOGL", "AMZN"])
    period_choice = st.selectbox("Historical Period", ["1 Year", "2 Years", "5 Years"], index=1)
    
    st.divider()
    st.subheader("Training Parameters")
    lookback = st.slider("Lookback Window (W days)", 10, 60, 30)
    horizon = st.slider("Forecast Horizon (N days)", 1, 15, 5)
    buy_threshold = st.slider("BUY Threshold (%)", 0.5, 5.0, 2.0) / 100
    
    st.divider()
    lstm_units = st.select_slider("LSTM Units", options=[32, 64, 128], value=64)
    dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2)
    
    run_btn = st.button("🚀 RUN COMPLETE ANALYSIS", use_container_width=True)

# --- FUNCTIONS ---

def get_data(ticker, period_str):
    years = int(period_str.split()[0])
    end = datetime.now()
    start = end - timedelta(days=years*365)
    df = yf.download(ticker, start=start, end=end)
    
    # Fix Multi-index column issue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df.index = df.index.tz_localize(None)
    return df

def engineer_features(df):
    df = df.copy()
    # Step 2: 10+ Required Features
    df['body_size'] = df['Close'] - df['Open']
    # Fixing division to avoid ValueError
    df['body_ratio'] = df['body_size'] / (df['High'] - df['Low']).replace(0, np.nan)
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['direction'] = np.where(df['Close'] > df['Open'], 1, -1)
    
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_5'] = df['Close'].pct_change(5)
    df['ret_10'] = df['Close'].pct_change(10)
    
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['rel_SMA_20'] = df['Close'] / df['SMA_20']
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['ATR'] = df['Close'].rolling(20).std()
    df['rel_vol'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df.dropna()

# --- MAIN EXECUTION ---
if run_btn:
    # 1. DATA COLLECTION
    df = get_data(ticker, period_choice)
    df_feat = engineer_features(df)
    
    # 2. LABELLING
    df_feat['Target'] = (df_feat['Close'].shift(-horizon) > df_feat['Close'] * (1 + buy_threshold)).astype(int)
    
    features = ['body_size', 'body_ratio', 'upper_shadow', 'lower_shadow', 'direction', 
                'ret_1', 'ret_5', 'ret_10', 'rel_SMA_20', 'MACD', 'ATR', 'rel_vol']
    
    X_raw = df_feat[features].values
    y_raw = df_feat['Target'].values[:-horizon]
    X_raw = X_raw[:-horizon]
    
    # 3. SPLITTING & SCALING
    train_size = int(len(X_raw) * 0.8)
    val_size = int(len(X_raw) * 0.1)
    
    scaler = MinMaxScaler()
    scaler.fit(X_raw[:train_size])
    X_scaled = scaler.transform(X_raw)
    
    X_windows, y_windows = [], []
    for i in range(lookback, len(X_scaled)):
        X_windows.append(X_scaled[i-lookback:i])
        y_windows.append(y_raw[i])
    
    X_windows, y_windows = np.array(X_windows), np.array(y_windows)
    
    X_train = X_windows[:train_size]
    y_train = y_windows[:train_size]
    X_val = X_windows[train_size:train_size+val_size]
    y_val = y_windows[train_size:train_size+val_size]
    X_test = X_windows[train_size+val_size:]
    y_test = y_windows[train_size+val_size:]
    
    # 4. MODEL ARCHITECTURE
    input_dim = len(features)
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(lookback, input_dim)),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 5. TRAINING
    with st.spinner('Training Stacked LSTM Model...'):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30, batch_size=32,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            ], verbose=0
        )

    # 6. RESULTS
    st.header(f"Results for {ticker}")
    
    latest_window = X_scaled[-lookback:].reshape(1, lookback, input_dim)
    prob = model.predict(latest_window)[0][0]
    
    if prob >= 0.65: decision, color, bg = "STRONG BUY", "#00FF00", "#002200"
    elif prob >= 0.52: decision, color, bg = "BUY", "#88FF88", "#113311"
    elif prob >= 0.35: decision, color, bg = "HOLD / WAIT", "#FFFF00", "#222200"
    else: decision, color, bg = "SELL", "#FF0000", "#440000"
    
    st.markdown(f"""
        <div class="decision-box" style="background-color:{bg}; color:{color}; border-color:{color};">
            <h1 style="margin:0;">{decision}</h1>
            <p style="font-size:20px; margin:0;">AI Prediction Confidence: {prob:.2%}</p>
        </div>
    """, unsafe_allow_html=True)

    # Metrics
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
    c2.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_prob):.2f}")
    c3.info("Powered by Stacked LSTM")

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'))
    fig.update_layout(title="Training History", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    st.success("Analysis Finished Successfully!")
else:
    st.info("Select parameters and click 'RUN ANALYSIS' to start.")
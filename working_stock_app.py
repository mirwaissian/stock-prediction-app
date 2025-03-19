import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import statsmodels.api as sm
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import ta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Advanced Stock Prediction", layout="wide")

# Header
st.title("Advanced Stock Price Prediction")
st.markdown("Analyze stocks using multiple ML models including neural networks")

# Set seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # Stock selection
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    
    # Date range
    st.subheader("Date Range")
    
    from datetime import datetime, timedelta
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365*5)  # Default to 5 years
    
    start_date = st.date_input("Start Date", start_date)
    end_date = st.date_input("End Date", end_date)
    
    # Model selection
    st.subheader("Models to Run")
    run_traditional = st.checkbox("Traditional Models", value=True, help="Ridge, Lasso, Random Forest, Gradient Boosting")
    run_neural = st.checkbox("Neural Networks", value=True, help="Dense NN, LSTM, CNN-LSTM")
    run_trading = st.checkbox("Trading Strategy", value=True)

# Function definitions from your code
def add_features(df):
    """Create comprehensive feature set for stock price prediction"""
    # Copy dataframe to avoid modifying original
    df_feat = df.copy()
    
    # Make sure we're working with Series, not DataFrames
    close_series = df_feat['Close'].squeeze()
    
    # 1. Price-based features
    df_feat['High_Low_Ratio'] = df_feat['High'] / df_feat['Low']
    df_feat['Close_Open_Ratio'] = df_feat['Close'] / df_feat['Open']
    
    # 2. Returns
    df_feat['Daily_Return'] = df_feat['Close'].pct_change()
    df_feat['Log_Return'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
    
    # 3. Moving Averages
    for window in [5, 10, 20, 50, 200]:
        df_feat[f'MA_{window}'] = close_series.rolling(window=window).mean()
    
    # 4. Volatility Indicators
    for window in [5, 20]:
        df_feat[f'Volatility_{window}'] = df_feat['Daily_Return'].rolling(window=window).std()
    
    # 5. Technical Indicators
    # RSI
    for window in [7, 14]:
        rsi = ta.momentum.RSIIndicator(close_series, window=window)
        df_feat[f'RSI_{window}'] = rsi.rsi()
    
    # MACD
    macd = ta.trend.MACD(close_series)
    df_feat['MACD'] = macd.macd()
    df_feat['MACD_Signal'] = macd.macd_signal()
    df_feat['MACD_Diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close_series, window=20)
    df_feat['BB_High'] = bollinger.bollinger_hband()
    df_feat['BB_Low'] = bollinger.bollinger_lband()
    df_feat['BB_Width'] = bollinger.bollinger_wband()
    
    # 6. Volume Indicators
    df_feat['Volume_Change'] = df_feat['Volume'].pct_change()
    df_feat['Volume_MA_5'] = df_feat['Volume'].rolling(window=5).mean()
    
    # On-Balance Volume (OBV)
    volume_series = df_feat['Volume'].squeeze()
    obv = ta.volume.OnBalanceVolumeIndicator(close_series, volume_series)
    df_feat['OBV'] = obv.on_balance_volume()
    df_feat['OBV_Change'] = df_feat['OBV'].pct_change()
    
    # 7. Target - Next day's closing price
    df_feat['Target'] = df_feat['Close'].shift(-1)
    
    # Drop NaN values
    df_feat.dropna(inplace=True)
    
    return df_feat

def create_sequences(X, y, time_steps=10):
    """Create sequences for time series models"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def evaluate_model(model_name, y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }

def evaluate_trading_strategy(y_true, y_pred, initial_capital=10000):
    """Evaluate a simple trading strategy based on predictions"""
    # Convert inputs to numpy arrays for consistent handling
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Make sure inputs are the same length
    if len(y_true_np) != len(y_pred_np):
        min_len = min(len(y_true_np), len(y_pred_np))
        st.warning(f"Length mismatch. y_true: {len(y_true_np)}, y_pred: {len(y_pred_np)}. Trimming to {min_len}.")
        y_true_np = y_true_np[:min_len]
        y_pred_np = y_pred_np[:min_len]
    
    # For calculation, we need to remove the last element from y_pred
    # and the first element from y_true to align properly
    n = len(y_true_np) - 1
    
    # Current day's price (n elements)
    current_price = y_true_np[:n]
    
    # Next day's actual price (n elements)
    next_price = y_true_np[1:n+1]
    
    # Today's prediction (n elements)
    prediction = y_pred_np[:n]
    
    # Signal: Buy if predicted price is higher than current price, else Sell
    signal = np.where(prediction > current_price, 1, -1)
    
    # Calculate returns
    actual_returns = (next_price / current_price - 1) * signal
    
    # Cumulative returns
    cumulative_returns = np.cumprod(1 + actual_returns)
    
    # Strategy performance
    total_return = cumulative_returns[-1] - 1
    annualized_return = ((1 + total_return) ** (252 / len(cumulative_returns)) - 1)
    sharpe_ratio = np.sqrt(252) * np.mean(actual_returns) / np.std(actual_returns)
    
    # Buy and hold strategy
    buy_hold_return = (next_price[-1] / current_price[0]) - 1
    
    # Plot equity curve
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If original data had an index (was a pandas Series)
    if hasattr(y_true, 'index'):
        idx = y_true.index[1:n+1]  # Use the same index as next_price
        ax.plot(idx, cumulative_returns * initial_capital, label='Strategy Equity')
        
        # Calculate buy & hold equity curve
        buy_hold_eq = initial_capital * (next_price / current_price[0])
        ax.plot(idx, buy_hold_eq, label='Buy & Hold Equity')
    else:
        # Simple plot without index
        ax.plot(cumulative_returns * initial_capital, label='Strategy Equity')
        buy_hold_eq = initial_capital * (next_price / current_price[0])
        ax.plot(buy_hold_eq, label='Buy & Hold Equity')
    
    ax.set_title('Trading Strategy Performance')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Buy & Hold Return': buy_hold_return
    }

# Main app execution
if st.button("Run Analysis", type="primary"):
    try:
        # Download data
        with st.spinner(f"Downloading {ticker} data..."):
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data found for {ticker} in the selected date range.")
            else:
                # Display stock info
                st.header(f"{ticker} Stock Analysis")
                
                # Basic stock chart
                st.subheader("Stock Price History")
                st.line_chart(data['Close'])
                
                # Display basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Current Price", 
                        f"${float(data['Close'].iloc[-1]):.2f}", 
                        f"{float((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):.2f}%"
                    )
                with col2:
                    st.metric(
                        "Average Volume", 
                        f"{int(data['Volume'].mean()):,}"
                    )
                with col3:
                    st.metric(
                        "Volatility (20d)", 
                        f"{float(data['Close'].pct_change().rolling(20).std().iloc[-1] * 100):.2f}%"
                    )
                
                # Feature Engineering
                with st.spinner("Calculating technical indicators..."):
                    data_with_features = add_features(data)
                    st.success(f"Added {len(data_with_features.columns) - len(data.columns)} technical indicators")
                
                # Show correlation matrix
                if st.checkbox("Show Correlation Matrix"):
                    st.subheader("Feature Correlations")
                    selected_cols = ['Close', 'Daily_Return', 'MA_20', 'RSI_14', 'MACD', 'BB_Width', 'OBV', 'Target']
                    corr = data_with_features[selected_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                    st.pyplot(fig)
                
                # Data Splitting and Scaling
                features = [col for col in data_with_features.columns if col not in ['Target']]
                X = data_with_features[features]
                y = data_with_features['Target']
                
                # Split data with 80% training, 20% testing (maintaining time order)
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                # Scale the data
                scaler_X = StandardScaler()
                X_train_scaled = scaler_X.fit_transform(X_train)
                X_test_scaled = scaler_X.transform(X_test)
                
                # For neural networks, also scale the target
                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
                
                # Model training and evaluation
                model_performances = []
                
                if run_traditional:
                    st.header("Traditional Models")
                    with st.spinner("Training traditional models..."):
                        # 1. Ridge Regression
                        ridge_model = Ridge(alpha=1.0)
                        ridge_model.fit(X_train_scaled, y_train)
                        ridge_preds = ridge_model.predict(X_test_scaled)
                        ridge_metrics = evaluate_model('Ridge Regression', y_test, ridge_preds)
                        model_performances.append(ridge_metrics)
                        
                        # Feature importance for Ridge
                        ridge_importance = np.abs(ridge_model.coef_)
                        
                        # 2. Lasso Regression
                        lasso_model = Lasso(alpha=0.01)
                        lasso_model.fit(X_train_scaled, y_train)
                        lasso_preds = lasso_model.predict(X_test_scaled)
                        lasso_metrics = evaluate_model('Lasso Regression', y_test, lasso_preds)
                        model_performances.append(lasso_metrics)
                        
                        # Feature importance for Lasso
                        lasso_importance = np.abs(lasso_model.coef_)
                        
                        # 3. Random Forest
                        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                        rf_model.fit(X_train_scaled, y_train)
                        rf_preds = rf_model.predict(X_test_scaled)
                        rf_metrics = evaluate_model('Random Forest', y_test, rf_preds)
                        model_performances.append(rf_metrics)
                        
                        # Feature importance for Random Forest
                        rf_importance = rf_model.feature_importances_
                        
                        # 4. Gradient Boosting
                        gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
                        gb_model.fit(X_train_scaled, y_train)
                        gb_preds = gb_model.predict(X_test_scaled)
                        gb_metrics = evaluate_model('Gradient Boosting', y_test, gb_preds)
                        model_performances.append(gb_metrics)
                        
                        # Feature importance for Gradient Boosting
                        gb_importance = gb_model.feature_importances_
                        
                        # Show feature importance for the best traditional model
                        st.subheader("Feature Importance")
                        trad_models = pd.DataFrame(model_performances)
                        best_trad_model = trad_models.loc[trad_models['RMSE'].idxmin(), 'Model']
                        
                        if best_trad_model == 'Ridge Regression':
                            importance = ridge_importance
                        elif best_trad_model == 'Lasso Regression':
                            importance = lasso_importance
                        elif best_trad_model == 'Random Forest':
                            importance = rf_importance
                        elif best_trad_model == 'Gradient Boosting':
                            importance = gb_importance
                        
                        feature_importance = pd.DataFrame({
                            'Feature': [str(f) for f in features],
                            'Importance': importance
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.barh(feature_importance['Feature'], feature_importance['Importance'])
                        ax.set_title(f'Feature Importance - {best_trad_model}')
                        ax.set_xlabel('Importance')
                        st.pyplot(fig)
                
                if run_neural:
                    st.header("Neural Network Models")
                    
                    # Prepare sequences for time series models
                    time_steps = 10
                    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
                    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)
                    
                    # Define callbacks
                    early_stopping = callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=15,
                        restore_best_weights=True
                    )
                    
                    reduce_lr = callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=5,
                        min_lr=0.00001
                    )
                    
                    # 1. Dense Neural Network
                    with st.spinner("Training Dense Neural Network..."):
                        dense_nn = keras.Sequential([
                            layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                            layers.BatchNormalization(),
                            layers.Dropout(0.3),
                            layers.Dense(64, activation='relu'),
                            layers.BatchNormalization(),
                            layers.Dropout(0.2),
                            layers.Dense(32, activation='relu'),
                            layers.Dense(1)
                        ])
                        
                        dense_nn.compile(
                            optimizer=optimizers.Adam(learning_rate=0.001),
                            loss='mse',
                            metrics=['mae']
                        )
                        
                        dense_history = dense_nn.fit(
                            X_train_scaled, y_train_scaled,
                            epochs=100,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[early_stopping, reduce_lr],
                            verbose=0
                        )
                        
                        # Plot training history
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        axes[0].plot(dense_history.history['loss'], label='Training Loss')
                        axes[0].plot(dense_history.history['val_loss'], label='Validation Loss')
                        axes[0].set_title('Dense NN - Loss')
                        axes[0].legend()
                        
                        axes[1].plot(dense_history.history['mae'], label='Training MAE')
                        axes[1].plot(dense_history.history['val_mae'], label='Validation MAE')
                        axes[1].set_title('Dense NN - MAE')
                        axes[1].legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Evaluate Dense NN
                        dense_preds_scaled = dense_nn.predict(X_test_scaled)
                        dense_preds = scaler_y.inverse_transform(dense_preds_scaled.reshape(-1, 1)).flatten()
                        dense_metrics = evaluate_model('Dense Neural Network', y_test, dense_preds)
                        model_performances.append(dense_metrics)
                    
                    # 2. LSTM Model
                    with st.spinner("Training LSTM Model..."):
                        lstm_model = keras.Sequential([
                            layers.LSTM(100, return_sequences=True, input_shape=(time_steps, X_train_scaled.shape[1])),
                            layers.Dropout(0.3),
                            layers.LSTM(50),
                            layers.Dropout(0.2),
                            layers.Dense(25, activation='relu'),
                            layers.Dense(1)
                        ])
                        
                        lstm_model.compile(
                            optimizer=optimizers.Adam(learning_rate=0.001),
                            loss='mse',
                            metrics=['mae']
                        )
                        
                        lstm_history = lstm_model.fit(
                            X_train_seq, y_train_seq,
                            epochs=100,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[early_stopping, reduce_lr],
                            verbose=0
                        )
                        
                        # Plot training history
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        axes[0].plot(lstm_history.history['loss'], label='Training Loss')
                        axes[0].plot(lstm_history.history['val_loss'], label='Validation Loss')
                        axes[0].set_title('LSTM - Loss')
                        axes[0].legend()
                        
                        axes[1].plot(lstm_history.history['mae'], label='Training MAE')
                        axes[1].plot(lstm_history.history['val_mae'], label='Validation MAE')
                        axes[1].set_title('LSTM - MAE')
                        axes[1].legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Evaluate LSTM
                        lstm_preds_scaled = lstm_model.predict(X_test_seq)
                        lstm_preds_full = scaler_y.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
                        
                        # Adjust dimensions to match y_test (due to sequence creation)
                        lstm_metrics = evaluate_model('LSTM', y_test[time_steps:], lstm_preds_full)
                        model_performances.append(lstm_metrics)
                    
                    # 3. CNN-LSTM Hybrid
                    with st.spinner("Training CNN-LSTM Hybrid Model..."):
                        cnn_lstm_model = keras.Sequential([
                            layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', 
                                         input_shape=(time_steps, X_train_scaled.shape[1])),
                            layers.MaxPooling1D(pool_size=2),
                            layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
                            layers.MaxPooling1D(pool_size=2),
                            layers.LSTM(50, return_sequences=False),
                            layers.Dropout(0.2),
                            layers.Dense(25, activation='relu'),
                            layers.Dense(1)
                        ])
                        
                        cnn_lstm_model.compile(
                            optimizer=optimizers.Adam(learning_rate=0.001),
                            loss='mse',
                            metrics=['mae']
                        )
                        
                        cnn_lstm_history = cnn_lstm_model.fit(
                            X_train_seq, y_train_seq,
                            epochs=100,
                            batch_size=32,
                            validation_split=0.2,
                            callbacks=[early_stopping, reduce_lr],
                            verbose=0
                        )
                        
                        # Plot training history
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        axes[0].plot(cnn_lstm_history.history['loss'], label='Training Loss')
                        axes[0].plot(cnn_lstm_history.history['val_loss'], label='Validation Loss')
                        axes[0].set_title('CNN-LSTM - Loss')
                        axes[0].legend()
                        
                        axes[1].plot(cnn_lstm_history.history['mae'], label='Training MAE')
                        axes[1].plot(cnn_lstm_history.history['val_mae'], label='Validation MAE')
                        axes[1].set_title('CNN-LSTM - MAE')
                        axes[1].legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Evaluate CNN-LSTM
                        cnn_lstm_preds_scaled = cnn_lstm_model.predict(X_test_seq)
                        cnn_lstm_preds = scaler_y.inverse_transform(cnn_lstm_preds_scaled.reshape(-1, 1)).flatten()
                        cnn_lstm_metrics = evaluate_model('CNN-LSTM Hybrid', y_test[time_steps:], cnn_lstm_preds)
                        model_performances.append(cnn_lstm_metrics)
                
                # Model Comparison
                st.header("Model Comparison")
                performance_df = pd.DataFrame(model_performances)
                st.dataframe(performance_df.style.highlight_min(subset=['RMSE', 'MAE', 'MAPE']).highlight_max(subset=['R²']))
                
                # Plot comparison of model performance
                metrics_to_plot = ['RMSE', 'MAE', 'R²']
                fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 12))
                
                for i, metric in enumerate(metrics_to_plot):
                    performance_df.sort_values(metric, ascending=False if metric == 'R²' else True).plot(
                        kind='barh', x='Model', y=metric, ax=axes[i], legend=False
                    )
                    axes[i].set_title(f'Model Comparison - {metric}')
                    axes[i].set_xlabel(metric)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Plot predictions for best model
                best_model_name = performance_df.loc[performance_df['RMSE'].idxmin(), 'Model']
                st.subheader(f"Predictions using {best_model_name}")
                
                # Get predictions for comparison
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot actual prices
                ax.plot(y_test.index[-60:], y_test.values[-60:], label='Actual', linewidth=2)
                
                # Plot the appropriate prediction line
                if best_model_name == 'LSTM':
                    ax.plot(y_test.index[-60:], lstm_preds_full[-60:], 'r--', label='LSTM', linewidth=1.5)
                elif best_model_name == 'CNN-LSTM Hybrid':
                    ax.plot(y_test.index[-60:], cnn_lstm_preds[-60:], 'g--', label='CNN-LSTM', linewidth=1.5)
                elif best_model_name == 'Dense Neural Network':
                    ax.plot(y_test.index[-60:], dense_preds[-60:], 'm--', label='Dense NN', linewidth=1.5)
                elif best_model_name == 'Ridge Regression':
                    ax.plot(y_test.index[-60:], ridge_preds[-60:], 'c--', label='Ridge', linewidth=1.5)
                elif best_model_name == 'Lasso Regression':
                    ax.plot(y_test.index[-60:], lasso_preds[-60:], 'y--', label='Lasso', linewidth=1.5)
                elif best_model_name == 'Random Forest':
                    ax.plot(y_test.index[-60:], rf_preds[-60:], 'k--', label='Random Forest', linewidth=1.5)
                elif best_model_name == 'Gradient Boosting':
                    ax.plot(y_test.index[-60:], gb_preds[-60:], 'b--', label='Gradient Boosting', linewidth=1.5)
                
                ax.set_title('Stock Price Predictions - Last 60 Days')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Trading Strategy
                if run_trading:
                    st.header("Trading Strategy Evaluation")
                    
                    # Get predictions from the best model
                    if best_model_name == 'LSTM':
                        strategy_metrics = evaluate_trading_strategy(y_test[time_steps:], lstm_preds_full)
                    elif best_model_name == 'CNN-LSTM Hybrid':
                        strategy_metrics = evaluate_trading_strategy(y_test[time_steps:], cnn_lstm_preds)
                    elif best_model_name == 'Dense Neural Network':
                        strategy_metrics = evaluate_trading_strategy(y_test, dense_preds)
                    elif best_model_name == 'Ridge Regression':
                        strategy_metrics = evaluate_trading_strategy(y_test, ridge_preds)
                    elif best_model_name == 'Lasso Regression':
                        strategy_metrics = evaluate_trading_strategy(y_test, lasso_preds)
                    elif best_model_name == 'Random Forest':
                        strategy_metrics = evaluate_trading_strategy(y_test, rf_preds)
                    elif best_model_name == 'Gradient Boosting':
                        strategy_metrics = evaluate_trading_strategy(y_test, gb_preds)
                    
                    # Display strategy metrics
                    st.subheader("Strategy Performance")
                    cols = st.columns(4)
                    cols[0].metric("Total Return", f"{strategy_metrics['Total Return']*100:.2f}%")
                    cols[1].metric("Annualized Return", f"{strategy_metrics['Annualized Return']*100:.2f}%")
                    cols[2].metric("Sharpe Ratio", f"{strategy_metrics['Sharpe Ratio']:.2f}")
                    cols[3].metric("Buy & Hold Return", f"{strategy_metrics['Buy & Hold Return']*100:.2f}%")
                
                # Final summary
                st.header("Summary")
                st.write(f"✅ Best model: **{best_model_name}** with RMSE of {performance_df.loc[performance_df['RMSE'].idxmin(), 'RMSE']:.4f}")
                
                if 'rf_importance' in locals():
                    top_features = pd.DataFrame({
                        'Feature': [str(f) for f in features],
                        'Importance': rf_importance
                    }).sort_values('Importance', ascending=False).head(5)
                    
                    st.write("**Top 5 Features (from Random Forest):**")
                    for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance']), 1):
                        st.write(f"{i}. {feature}: {importance:.4f}")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)
else:
    # Display welcome message
    st.info("👈 Configure analysis parameters in the sidebar and click 'Run Analysis'")
    
    # Show sample image
    st.image("https://static.streamlit.io/examples/stock.jpg", width=600)
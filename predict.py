import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split  # Corrected: Use train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns  # For better visualizations

# --- 1. Data Acquisition and Feature Engineering ---

def get_and_preprocess_data(ticker, start_date, end_date, lookback_period=30):
    """
    Fetches data, preprocesses, and engineers features.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        lookback_period (int): Number of past days to use for prediction.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler, label_encoder)
               or (None, None, None, None, None, None) if there's an error.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print(f"No data found for ticker: {ticker}")
            return None, None, None, None, None, None

        # --- Feature Engineering (Beyond Basic OHLCV) ---

        # 1. Price Ratios (avoiding TA-Lib's typical indicators)
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['High_Low_Ratio'] = df['High'] / df['Low']

        # 2. Rolling Statistics (Custom, not SMA/EMA)
        df['Rolling_Mean_Volume'] = df['Volume'].rolling(window=5).mean()
        df['Rolling_Std_Close'] = df['Close'].rolling(window=5).std()

        # 3. Lagged Returns (Percentage Change)
        for i in range(1, lookback_period + 1):
            df[f'Return_{i}'] = df['Close'].pct_change(periods=i)

        # 4. Volume-Weighted Average Price (VWAP) - Custom Implementation
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        # 5. Price Momentum (Rate of Change, without RSI)
        df['Momentum'] = df['Close'] - df['Close'].shift(5)

        # 6.  Day of Week/Month (Cyclical Features) - IMPORTANT for time series.
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month

        # --- Target Variable: Price Movement (Classification) ---
        df['Price_Movement'] = np.where(df['Close'].shift(-1) > df['Close'], 1,  # Up
                                        np.where(df['Close'].shift(-1) < df['Close'], 0, 2))  # Down, Neutral

        # --- Data Cleaning and Preprocessing ---
        df = df.dropna()  # Drop rows with NaN values (created by rolling features, etc.)

        # Separate features (X) and target (y)
        X = df.drop(columns=['Price_Movement'])
        y = df['Price_Movement']

        # Feature Scaling (MinMaxScaler for numerical features)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Label Encoding for the target variable (0, 1, 2)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # --- Reshape for LSTM: [samples, timesteps, features] ---
        X_reshaped = []
        y_reshaped = []
        for i in range(lookback_period, len(X_scaled)):
            X_reshaped.append(X_scaled[i-lookback_period:i])
            y_reshaped.append(y_encoded[i])  # Use encoded y

        X_reshaped, y_reshaped = np.array(X_reshaped), np.array(y_reshaped)

        # --- Train/Test Split (Time-Series Aware) ---
        #  Avoid TimeSeriesSplit for *classification* with a discrete split point.
        test_size = 0.2
        split_index = int(len(X_reshaped) * (1 - test_size))
        X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        y_train, y_test = y_reshaped[:split_index], y_reshaped[split_index:]

        return X_train, X_test, y_train, y_test, scaler, label_encoder

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None



# --- 2. Model Building (LSTM) ---

def build_lstm_model(input_shape):
    """
    Builds and compiles the LSTM model.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
        keras.Model: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape, activation='tanh')) #tanh often better for LSTMs.
    model.add(Dropout(0.2))  # Regularization
    model.add(LSTM(units=32, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(units=16, activation='relu')) # ReLU for hidden
    model.add(Dense(units=3, activation='softmax'))  # 3 output units for classes (up, down, neutral)

    optimizer = Adam(learning_rate=0.001)  # Adjustable learning rate.
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# --- 3. Model Training and Evaluation ---

def train_and_evaluate_model(X_train, X_test, y_train, y_test, epochs=50, batch_size=32):
    """
    Trains and evaluates the LSTM model.

    Args:
        X_train, X_test, y_train, y_test: Training and testing data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        tuple: (trained_model, history)
    """
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # --- Callbacks for Better Training ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stopping, reduce_lr], verbose=1)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Down', 'Up', 'Neutral'], yticklabels=['Down', 'Up', 'Neutral'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


    # --- Plot Training History ---
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.show()


    return model, history

# --- 4. Prediction Function ---

def predict_movement(model, scaler, label_encoder, data, lookback_period):
    """
    Predicts the next day's price movement.

    Args:
    model: Trained Keras model
    scaler: Fitted MinMaxScaler
    label_encoder: Fitted LabelEncoder
    data:  DataFrame with the most recent data.  Must contain the SAME columns used during training.
    lookback_period: lookback period used during training.

    Returns:
    str: Predicted price movement ('Up', 'Down', or 'Neutral').
    """
    # Preprocess the 'data' DataFrame in the *EXACT* same way as the training data.
    # This includes feature engineering.
    data['Open_Close_Ratio'] = data['Open'] / data['Close']
    data['High_Low_Ratio'] = data['High'] / data['Low']
    data['Rolling_Mean_Volume'] = data['Volume'].rolling(window=5).mean()
    data['Rolling_Std_Close'] = data['Close'].rolling(window=5).std()
    for i in range(1, lookback_period + 1):
        data[f'Return_{i}'] = data['Close'].pct_change(periods=i)
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data = data.dropna() # Drop any NaNs.

    # Extract the last 'lookback_period' rows.
    recent_data = data.iloc[-lookback_period:]

    if len(recent_data) < lookback_period:
        raise ValueError("Not enough data for prediction.  Need at least 'lookback_period' days.")

    # Scale the features using the *pre-fitted* scaler.
    scaled_recent_data = scaler.transform(recent_data)

    # Reshape for LSTM: [samples, timesteps, features]
    recent_data_reshaped = np.array([scaled_recent_data]) # Add an extra dimension.

    # Make the prediction.
    prediction_probs = model.predict(recent_data_reshaped)
    predicted_class = np.argmax(prediction_probs, axis=1)[0]  # Get the class with highest probability.

    # Inverse transform to get the original label ('Up', 'Down', 'Neutral')
    predicted_movement = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_movement


# --- 5. Main Execution ---

if __name__ == "__main__":
    ticker = 'MSFT'
    start_date = '2018-01-01'
    end_date = '2023-10-26'
    lookback_period = 30
    epochs = 50
    batch_size = 32

    X_train, X_test, y_train, y_test, scaler, label_encoder = get_and_preprocess_data(
        ticker, start_date, end_date, lookback_period
    )

    if X_train is not None:  # Check if data loading was successful
        model, history = train_and_evaluate_model(X_train, X_test, y_train, y_test, epochs, batch_size)

        # --- Predict Next Day's Movement ---
        # Get the *most recent* data (including today, if possible).
        today = pd.Timestamp.today().strftime('%Y-%m-%d')
        recent_data = yf.download(ticker, start=pd.Timestamp.today() - pd.DateOffset(days=lookback_period*2), end=today) # Get extra data

        predicted_movement = predict_movement(model, scaler, label_encoder, recent_data, lookback_period)
        print(f"Predicted next day's movement for {ticker}: {predicted_movement}")

    else:
        print("Data loading failed.  Cannot proceed.")
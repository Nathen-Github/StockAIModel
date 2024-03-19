import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Convert date column to datetime and set it as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Drop the ticker symbol column
    df = df.drop(columns=[df.columns[0]])

    # Assuming that the 'close' column is the last column and all others are features
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]
    
    # Scaling features and target
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

    print("Features and target scaled successfully.")
    return scaled_features, scaled_target, scaler

def create_sequences(features, target, sequence_length=60):
    print("Creating sequences")
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(target[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    print("Building the LSTM model")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


if __name__ == "__main__":
    filepath = '/Users/vadakkemury/desktop/AI_model/stock_data/tsla/TSLA_merged_yearly_data.csv'
    scaled_features, scaled_target, scaler = load_and_preprocess_data(filepath)
    X, y = create_sequences(scaled_features, scaled_target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    print("Starting model training")
    model.fit(X_train, y_train, epochs=50, batch_size=20, validation_split=0.1)
    
    print("Evaluating model")
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate Percent Error for each prediction
    percent_errors = np.abs((real_prices - predicted_prices) / real_prices) * 100
    mean_percent_error = np.mean(percent_errors)
    print(f"Mean Percent Error: {mean_percent_error}%")

    # Calculate changes in actual and predicted prices to determine direction
    actual_changes = np.diff(real_prices.reshape(-1))  # Flatten array and calculate differences
    predicted_changes = np.diff(predicted_prices.reshape(-1))  # Flatten array and calculate differences

    # Determine if the direction of change matches (actual vs. predicted)
    direction_matches = np.sign(actual_changes) == np.sign(predicted_changes)
    
    # Calculate directional accuracy
    directional_accuracy = np.mean(direction_matches) * 100  # Convert to percentage
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")

    # Continue with plotting and other evaluations as before
    plt.figure(figsize=(10, 6))
    plt.plot(real_prices, label='Actual Prices')
    plt.plot(predicted_prices, label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    print("Process completed with new testing data.")
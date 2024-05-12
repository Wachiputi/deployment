import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
import datetime
from pathlib import Path

# Set title and sidebar info
st.title('Agricultural Commodity Price Projection')
st.sidebar.info('Welcome to the Agricultural Commodity Price Projection App. Choose your options below')

# Sidebar options
option = st.sidebar.selectbox('Select Crop', ['Maize (new harvest)', 'Beans'])
start_date = st.sidebar.date_input('Start Date', value=datetime.date.today() - datetime.timedelta(days=365))
end_date = st.sidebar.date_input('End Date', datetime.date.today())
num_days_forecast = st.sidebar.number_input('Number of days to forecast', value=7, min_value=1, max_value=365, step=1)
selected_district = st.sidebar.selectbox('Select District', ['Dedza', 'Mzimba', 'Blantyre', 'Ntcheu'])

# Filter markets based on selected district
if selected_district == 'Dedza':
    markets = ['Nsikawanjala']
elif selected_district == 'Mzimba':
    markets = ['Jenda']
elif selected_district == 'Blantyre':
    markets = ['Lunzu']
elif selected_district == 'Ntcheu':
    markets = ['Lizulu']
elif selected_district == 'Dowa':
    markets = ['Nsungwi']  # Add the markets for Ntcheu here

selected_market = st.sidebar.selectbox('Select Market', markets)
forecast_button = st.sidebar.button('Predict')

# Define the path to the CSV file
DATA_PATH = Path.cwd() / 'data' / 'wfp_food_prices_mwi.csv'

# Read the data from the CSV file
data = pd.read_csv(DATA_PATH)

# Display the raw data using Streamlit
st.subheader("Raw WFP Data")
st.write(data)

# Remove null values by replacing with bfill
ft_data = data.fillna('bfill', inplace=False)
ft_data.isnull().sum()

# Display the filtered data after filling null values
st.subheader("Filtered WFP Data (Nulls Filled)")
st.write(ft_data)

# Drop the specified columns
columns_to_drop = ['usdprice', 'latitude', 'longitude', 'category', 'unit', 'priceflag', 'currency', 'pricetype']
ft_data.drop(columns=columns_to_drop, inplace=True)
ft_data.drop(index=0, inplace=True)

# Display the data after dropping columns
st.subheader('Filtered Data After Dropping Columns')
st.write(ft_data)

# Filter data based on the date, commodity, district, and market
# Converting the date column to datetime format
ft_data['date'] = pd.to_datetime(ft_data['date'])

# Defining the date range
start_dates = start_date.strftime('%Y-%m-%d')
end_dates = end_date.strftime('%Y-%m-%d')

# Filtering the data for the date range, commodity, district, and market
filtered_df = ft_data[(ft_data['date'] >= start_dates) & (ft_data['date'] <= end_dates)]
filtered_df = filtered_df[(filtered_df['commodity'] == option) & (filtered_df['district'] == selected_district) & (filtered_df['market'] == selected_market)]

# Display the fully filtered data
st.subheader('Fully Filtered Data')
st.write(filtered_df)

# Generate trend graph for the fully filtered data
if not filtered_df.empty:
    fig, ax = plt.subplots()
    ax.plot(filtered_df['date'], filtered_df['price'], label='Historical Prices', marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Historical Prices Trend')
    ax.legend()
    st.pyplot(fig)

if forecast_button:
    # Filter the DataFrame based on the selected crop
    filtered_data = filtered_df[filtered_df['commodity'] == option]

    # Prepare the data for LSTM
    def prepare_data_for_lstm(data):
        # Convert price column to array
        prices = data['price'].values

        # Normalize prices
        scaler = StandardScaler()
        prices_normalized = scaler.fit_transform(prices.reshape(-1, 1))

        # Create sequences and corresponding targets
        sequences = []
        targets = []
        seq_length = 10  # Adjust the sequence length as needed
        for i in range(len(prices_normalized) - seq_length):
            sequences.append(prices_normalized[i:i+seq_length])
            targets.append(prices_normalized[i+seq_length])

        return np.array(sequences), np.array(targets)

    # Prepare data for LSTM
    X, y = prepare_data_for_lstm(filtered_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the LSTM model
    model = Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save("lstm_model.h5")

    # Evaluate the model
    mse = model.evaluate(X_test, y_test)
    st.write(f'Mean Squared Error: {mse}')

    # Forecasting for multiple days
    forecast_prices_normalized = model.predict(X_test)
    forecast_prices = scaler.inverse_transform(forecast_prices_normalized)

    # Plot historical and forecasted prices using a line plot
    fig, ax = plt.subplots()
    ax.plot(filtered_data.index[-len(y_test):], filtered_data['price'].values[-len(y_test):], label='Historical Prices', marker='o')
    ax.plot(filtered_data.index[-len(y_test):], forecast_prices, label='Forecasted Prices', marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Historical and Forecasted Prices')
    ax.legend()

    st.pyplot(fig)

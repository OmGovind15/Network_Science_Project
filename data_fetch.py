import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
from tqdm import tqdm # Import tqdm

# --- 1. Fetch the list of Nifty 500 stock symbols ---
csv_url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}
print(f"Downloading stock list from {csv_url}...")

try:
    response = requests.get(csv_url, headers=headers)
    response.raise_for_status() 
    csv_data = io.StringIO(response.text)
    nifty500_df = pd.read_csv(csv_data)
    symbols = nifty500_df['Symbol'].tolist()
    symbols_ns = [symbol + ".NS" for symbol in symbols]
    print(f"Successfully fetched {len(symbols_ns)} stock symbols.")
except requests.exceptions.RequestException as e:
    print(f"Error: Failed to download CSV file. {e}")
    symbols_ns = []

# --- 2. Download historical data (one by one) using yfinance and tqdm ---

if symbols_ns:
    start_date = "2023-01-01"
    end_date = "2024-12-31" 

    print(f"Downloading historical data for {len(symbols_ns)} stocks from {start_date} to {end_date}...")

    # This list will hold all the successful data Series
    all_price_data = []
    
    # Wrap the symbol list with tqdm to create a progress bar
    for symbol in tqdm(symbols_ns, desc="Downloading Data"):
        try:
            # Download data for one symbol at a time
            data = yf.download(symbol, 
                               start=start_date, 
                               end=end_date,
                               auto_adjust=True, # Added to suppress yfinance warning
                               progress=False) # Turn off yfinance's built-in progress bar
            
            if not data.empty:
                # We only care about the 'Close' price (since auto_adjust=True, it's already adjusted)
                price_series = data['Close']
                price_series.name = symbol # Rename the series to its symbol
                all_price_data.append(price_series)
            else:
                print(f"No data returned for {symbol}")

        except Exception as e:
            # If a download fails, print the error and continue
            print(f"\nFailed to get ticker '{symbol}' reason: {e}")

    # --- 3. Combine, Clean, and Save the Data ---
    if all_price_data:
        # Combine all the individual Series into one large DataFrame
        adj_close_prices = pd.concat(all_price_data, axis=1)

        # Drop any columns (stocks) that have ANY missing data (NaN)
        cleaned_prices = adj_close_prices.dropna(axis=1)
        
        # Save the cleaned data to a new local file
        local_filename = "nifty500_adj_close_2023_2024.csv"
        cleaned_prices.to_csv(local_filename)
        
        print(f"\nSuccessfully downloaded and combined data.")
        print(f"Original number of symbols requested: {len(symbols_ns)}")
        print(f"Successfully downloaded data for: {len(all_price_data)} stocks")
        print(f"Number of stocks after cleaning (no missing data): {len(cleaned_prices.columns)}")
        print(f"Data saved to '{local_filename}'")
    else:
        print("No data was successfully downloaded.")
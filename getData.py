# This script fetches historical stock data from the Alpaca API for the symbol "SPY" 
# at 1-minute intervals, resamples it to 30-minute intervals, and saves it as a CSV file.

# To use this code you will need to fill in:
    # - API KEY
    # - API SECRET
    # - BASE URL
# All of this very easy to find in the alpaca website, you just need your own

# I also added a small dataset under "Data.csv" in this folder so you don't have to worry about this if you don;t want to

import pandas as pd
from alpaca_trade_api import REST, TimeFrame
from datetime import datetime, timedelta

API_KEY = ""
API_SECRET = ""
BASE_URL = ""

# Initialize Alpaca API
api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

# Define symbol and date range
symbol = "SPY"
end_date = datetime.now()
start_date = end_date - timedelta(days=90)  # Past 4 months

# Fetch data at 1-minute intervals and then resample to 30-minute intervals
bars = api.get_bars(symbol, TimeFrame.Minute, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), adjustment='all').df
bars.index = pd.to_datetime(bars.index)
bars = bars.resample('30T').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
bars.dropna(inplace=True)

# Display the middle 20 rows of the data
print(bars.iloc[len(bars) // 2 - 10: len(bars) // 2 + 10])

# Save to CSV
bars.to_csv(r"")
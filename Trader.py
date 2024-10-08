#Cross validation. no Stop loss. no Profit loss

import pandas as pd
import numpy as np
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST, TimeFrame
from timedelta import Timedelta
from getSentiment import estimate_sentiment
import ta
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import logging

API_KEY = ""
API_SECRET = ""
BASE_URL = ""

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

# Configurable variables
SYMBOLS = ["SPY", "NVDA"]
CASH_AT_RISK = 0.6

class MLTrader(Strategy):
    def initialize(self, symbols: list = SYMBOLS, cash_at_risk: float = CASH_AT_RISK):
        self.symbols = symbols
        self.sleeptime = "12H"  # Increase trading frequency
        self.last_trade = {symbol: None for symbol in symbols}
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def position_sizing(self, symbol):
        cash = self.get_cash()
        last_price = self.get_last_price(symbol)
        bars = self.get_technical_indicators(symbol)
        atr = ta.volatility.AverageTrueRange(high=bars['high'], low=bars['low'], close=bars['close']).average_true_range().iloc[-1]
        quantity = max(1, round((cash * self.cash_at_risk) / (atr * last_price), 0))
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self, symbol):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def get_technical_indicators(self, symbol):
        bars = self.api.get_bars(symbol, TimeFrame.Day, start="2022-01-01", end="2024-05-27", limit=150).df
        bars.index = pd.to_datetime(bars.index)
        ichimoku = ta.trend.IchimokuIndicator(high=bars['high'], low=bars['low'])
        bars['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
        bars['kijun_sen'] = ichimoku.ichimoku_base_line()
        bars['senkou_span_a'] = ichimoku.ichimoku_a()
        bars['senkou_span_b'] = ichimoku.ichimoku_b()
        bars['rsi'] = ta.momentum.RSIIndicator(close=bars['close']).rsi()
        macd = ta.trend.MACD(close=bars['close'])
        bars['macd'] = macd.macd()
        bars['macd_signal'] = macd.macd_signal()
        return bars

    def on_trading_iteration(self):
        for symbol in self.symbols:
            self.logger.info(f"Processing {symbol}")
            cash, last_price, quantity = self.position_sizing(symbol)
            self.logger.info(f"Position size for {symbol}: Cash: {cash}, Last Price: {last_price}, Quantity: {quantity}")
            probability, sentiment = self.get_sentiment(symbol)
            self.logger.info(f"Sentiment for {symbol}: Probability: {probability}, Sentiment: {sentiment}")
            df = self.get_technical_indicators(symbol)
            latest = df.iloc[-1]

            self.logger.info(f"Technical indicators for {symbol}: {latest.to_dict()}")

            weighted_rsi = latest['rsi'] * self.parameters['weight_rsi']
            weighted_macd = (latest['macd'] - latest['macd_signal']) * self.parameters['weight_macd']
            weighted_ichimoku = ((latest['tenkan_sen'] - latest['kijun_sen']) +
                                 (latest['close'] - latest['senkou_span_a'])) * self.parameters['weight_ichimoku']

            composite_score = weighted_rsi + weighted_macd + weighted_ichimoku
            self.logger.info(f"Composite Score: {composite_score}")

            current_positions = self.get_positions()
            self.logger.info(f"Current positions for {symbol}: {current_positions}")
            
            if sentiment == "positive" and probability > .85 and latest['rsi'] < 65:
                if composite_score > 0 and symbol not in current_positions:
                    order = self.create_order(
                        symbol,
                        quantity,
                        "buy",
                        type="market"
                    )
                    self.submit_order(order)
                    self.logger.info(f"Created buy order for {symbol}")
                    self.last_trade[symbol] = "buy"
                elif composite_score < 0 and symbol in current_positions:
                    self.sell_all()
                    self.logger.info(f"Sold all positions for {symbol}")
                    self.last_trade[symbol] = "sell"
            elif sentiment == "negative" and probability > .85 and latest['rsi'] > 35:
                if composite_score < 0 and symbol not in current_positions:
                    order = self.create_order(
                        symbol,
                        quantity,
                        "sell",
                        type="market"
                    )
                    self.submit_order(order)
                    self.logger.info(f"Created sell order for {symbol}")
                    self.last_trade[symbol] = "sell"
                elif composite_score > 0 and symbol in current_positions:
                    self.sell_all()
                    self.logger.info(f"Sold all positions for {symbol}")
                    self.last_trade[symbol] = "buy"

            self.logger.info(f"Portfolio Val: {self.get_portfolio_value()}")

    def on_finish(self):
        self.sell_all()

def cross_validate_weights(strategy_class, symbols, cash_at_risk, start_date, end_date):
    all_results = []
    param_grid = {
        'weight_rsi': [0.2, 0.4, 0.6, 0.8, 1.0],
        'weight_macd': [0.2, 0.4, 0.6, 0.8, 1.0],
        'weight_ichimoku': [0.2, 0.4, 0.6, 0.8, 1.0]
    }
    param_combinations = list(ParameterGrid(param_grid))
    for params in param_combinations:
        print(f"Testing combination: {params}")
        strategy = strategy_class(
            name='mlstrat',
            broker=Alpaca(ALPACA_CREDS),
            parameters={"symbols": symbols, "cash_at_risk": cash_at_risk, **params}
        )
        results = strategy.backtest(
            YahooDataBacktesting,
            start_date,
            end_date,
            parameters={"symbols": symbols, "cash_at_risk": cash_at_risk, **params}
        )
        if not results.empty:
            total_return = results['total_return'].iloc[-1]
            max_drawdown = results['max_drawdown'].min()
            sharpe_ratio = results['sharpe_ratio'].iloc[-1]
            volatility = results['volatility'].iloc[-1]
            print(f"Results for combination {params}: Total Return: {total_return}, Max Drawdown: {max_drawdown}, Sharpe Ratio: {sharpe_ratio}, Volatility: {volatility}")
            all_results.append({
                'params': params,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility
            })
        else:
            print(f"No results for combination {params}")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("volatility_accuracy.csv")
    return results_df

# Define the testing period
start_date = datetime(2023, 5, 27)
end_date = datetime(2024, 5, 27)

# Run cross-validation
results = cross_validate_weights(MLTrader, SYMBOLS, CASH_AT_RISK, start_date, end_date)
print("Cross-validatcompleted.")
print(results)